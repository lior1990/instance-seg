import torch.nn as nn
import torch
import config
import numpy as np


class SingleClusterNet(nn.Module):
    '''
    This class is a clustering network that takes in an (Nx1x224x224) batches of "images" ("probabilities" to belong in the cluster)
    and outputs a clustered image. This model is based on U-Net architecture
    (U-Net: Convolutional Networks for Biomedical Image Segmentation)
    '''

    def __init__(self, useSkip, segmentWeight):
        super(SingleClusterNet, self).__init__()
        h = 224  # image dimensions currently hard-coded
        w = 224  # image dimensions currently hard-coded
        self.useSkip = useSkip
        inputDim = 1
        # dim is (inputDim)x224x224
        ds1OutDim = 64
        self.ds1 = DownSamplingBlock(3, inputDim, ds1OutDim)
        # dim is (ds1OutDim)x112x112
        ds2OutDim = 2 * ds1OutDim
        self.ds2 = DownSamplingBlock(3, ds1OutDim, ds2OutDim)
        # dim is (2ds1OutDim)x56x56
        ds3OutDim = 2 * ds2OutDim
        self.ds3 = DownSamplingBlock(3, ds2OutDim, ds3OutDim)
        # dim is (4ds1OutDim)x28x28
        ds4OutDim = ds3OutDim * 2
        self.ds4 = DownSamplingBlock(3, ds3OutDim, ds4OutDim)
        # dim is (8ds1OutDim)x14x14

        lowetConvOutDim = 2 * ds4OutDim
        self.lowestLevel = nn.Sequential(
            nn.Conv2d(in_channels=ds4OutDim, out_channels=lowetConvOutDim, kernel_size=3, padding=3 // 2),
            nn.BatchNorm2d(lowetConvOutDim),
            nn.ReLU(),
            nn.Conv2d(in_channels=lowetConvOutDim, out_channels=lowetConvOutDim, kernel_size=3, padding=3 // 2),
            nn.BatchNorm2d(lowetConvOutDim),
            nn.ReLU(),
        )
        # dim is (16ds1OutDim)x14x14

        self.us4 = UpSamplingBlock(3, lowetConvOutDim, ds4OutDim, useSkip)
        # dim is (8ds1OutDim)x28x28
        self.us3 = UpSamplingBlock(3, ds4OutDim, ds3OutDim, useSkip)
        # dim is (4ds1OutDim)x56x56
        self.us2 = UpSamplingBlock(3, ds3OutDim, ds2OutDim, useSkip)
        # dim is (2ds1OutDim)x112x112
        self.us1 = UpSamplingBlock(3, ds2OutDim, ds1OutDim, useSkip)
        # dim is (ds1OutDim)x224x224

        self.lastConv = nn.Conv2d(in_channels=ds1OutDim, out_channels=1, kernel_size=1)
        # dim is 1x224x224
        weight = torch.Tensor([segmentWeight]).type(config.float_type)
        self.loss = nn.BCEWithLogitsLoss(pos_weight=weight)  # reduction is mean

    def forward(self, diffsBatch, labelsBatch):
        batchSize = diffsBatch.shape[0]
        in1 = diffsBatch.type(config.float_type)
        # dim is (inputDim)x224x224
        out1, skip1 = self.ds1(in1)
        # dim is (ds1OutDim)x112x112
        out2, skip2 = self.ds2(out1)
        # dim is (2ds1OutDim)x56x56
        out3, skip3 = self.ds3(out2)
        # dim is (4ds1OutDim)x28x28
        out4, skip4 = self.ds4(out3)
        # dim is (8ds1OutDim)x14x14

        predictions = self.lowestLevel(out4)
        # dim is (16ds1OutDim)x14x14

        # dim is (16ds1OutDim)x14x14
        predictions = self.us4(predictions, skip4)
        # dim is (8ds1OutDim)x28x28
        predictions = self.us3(predictions, skip3)
        # dim is (4ds1OutDim)x56x56
        predictions = self.us2(predictions, skip2)
        # dim is (2ds1OutDim)x112x112
        predictions = self.us1(predictions, skip1)
        # dim is (ds1OutDim)x224x224
        predictions = self.lastConv(predictions)
        # dim is 1x224x224
        decisions = np.zeros(predictions.shape)
        decisions[np.where(predictions.detach().cpu().numpy() > 0)] = 1  # creating the detected mask
        totalLoss = None
        if self.training:
            totalLoss = self.loss(predictions, torch.from_numpy(labelsBatch).type(config.float_type))
            totalLoss = totalLoss * batchSize  # averaging in the outside loop, maybe the training is on multiple GPUs
        return decisions, totalLoss


class DownSamplingBlock(nn.Module):
    def __init__(self, kernelSize, inputFeatures, outputFeatures, factor=2):
        super(DownSamplingBlock, self).__init__()
        self.convLayer1 = nn.Conv2d(in_channels=inputFeatures, out_channels=outputFeatures, kernel_size=kernelSize,
                                    padding=kernelSize // 2)
        self.bn1 = nn.BatchNorm2d(num_features=outputFeatures)
        self.relu1 = nn.ReLU()
        self.convLayer2 = nn.Conv2d(in_channels=outputFeatures, out_channels=outputFeatures, kernel_size=kernelSize,
                                    padding=kernelSize // 2)
        self.bn2 = nn.BatchNorm2d(num_features=outputFeatures)
        self.pool = nn.MaxPool2d(kernel_size=factor)
        self.relu2 = nn.ReLU()

    def forward(self, inFeatures):
        outFeatures = self.convLayer1(inFeatures)
        outFeatures = self.bn1(outFeatures)
        outFeatures = self.relu1(outFeatures)
        outFeatures = self.convLayer2(outFeatures)
        outFeatures = self.bn2(outFeatures)
        outFeatures = self.relu2(outFeatures)
        outputForSkipConnection = outFeatures
        outFeatures = self.pool(outFeatures)
        return outFeatures, outputForSkipConnection


class UpSamplingBlock(nn.Module):
    def __init__(self, kernelSize, inputFeatures, outputFeatures, skipConnection, factor=2):
        super(UpSamplingBlock, self).__init__()
        self.skipConnection = skipConnection

        self.upsample = nn.Upsample(scale_factor=factor,mode='bilinear',align_corners=True)
        self.conv1 = nn.Conv2d(in_channels=inputFeatures, out_channels=outputFeatures,
                               kernel_size=kernelSize, padding=kernelSize // 2)  # smooth the upsample
        self.bn1 = nn.BatchNorm2d(outputFeatures)
        self.relu1 = nn.ReLU()

        conv2InFeatures = outputFeatures
        if skipConnection:
            conv2InFeatures *= 2
        self.conv2 = nn.Conv2d(in_channels=conv2InFeatures, out_channels=outputFeatures, kernel_size=kernelSize,
                               padding=kernelSize // 2)
        self.bn2 = nn.BatchNorm2d(outputFeatures)

        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=outputFeatures, out_channels=outputFeatures,
                               kernel_size=kernelSize, padding=kernelSize // 2)
        self.bn3 = nn.BatchNorm2d(outputFeatures)
        self.relu3 = nn.ReLU()

    def forward(self, inFeatures, skipInput):
        outFeatures = self.upsample(inFeatures)

        outFeatures = self.conv1(outFeatures)
        outFeatures = self.bn1(outFeatures)
        outFeatures = self.relu1(outFeatures)

        if self.skipConnection:
            outFeatures = torch.cat((outFeatures, skipInput), dim=1)
        outFeatures = self.conv2(outFeatures)
        outFeatures = self.bn2(outFeatures)
        outFeatures = self.relu2(outFeatures)

        outFeatures = self.conv3(outFeatures)
        outFeatures = self.bn3(outFeatures)
        outFeatures = self.relu3(outFeatures)
        return outFeatures
