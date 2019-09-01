import torch.nn as nn
import torch
import config
import numpy as np


class SingleClusterNet(nn.Module):
    def __init__(self, useSkip, segmentWeight):
        super(SingleClusterNet, self).__init__()
        h = 224
        w = 224
        self.useSkip = useSkip
        inputDim = 1
        # dim is (inputDim)x224x224
        ds1OutDim = 32
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
        ds5OutDim = ds4OutDim * 2
        self.ds5 = DownSamplingBlock(3, ds4OutDim, ds5OutDim)
        # dim is (16ds1OutDim)x7x7
        lowetConvOutDim = 2 * ds5OutDim
        self.lowestLevel = nn.Sequential(
            nn.Conv2d(in_channels=ds5OutDim, out_channels=lowetConvOutDim, kernel_size=3, padding=3 // 2),
            nn.BatchNorm2d(lowetConvOutDim),
            nn.ReLU(),
            nn.Conv2d(in_channels=lowetConvOutDim, out_channels=lowetConvOutDim, kernel_size=3, padding=3 // 2),
            nn.BatchNorm2d(lowetConvOutDim),
            nn.ReLU(),
        )
        # dim is (32ds1OutDim)x7x7
        self.us5 = UpSamplingBlock(3, lowetConvOutDim, ds5OutDim, useSkip)
        # dim is (16ds1OutDim)x14x14
        self.us4 = UpSamplingBlock(3, ds5OutDim, ds4OutDim, useSkip)
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
        out1, skip1, idx1 = self.ds1(in1)
        # dim is (ds1OutDim)x112x112
        out2, skip2, idx2 = self.ds2(out1)
        # dim is (2ds1OutDim)x56x56
        out3, skip3, idx3 = self.ds3(out2)
        # dim is (4ds1OutDim)x28x28
        out4, skip4, idx4 = self.ds4(out3)
        # dim is (8ds1OutDim)x14x14
        out5, skip5, idx5 = self.ds5(out4)
        # dim is (16ds1OutDim)x7x7

        predictions = self.lowestLevel(out5)
        # dim is (32ds1OutDim)x7x7

        predictions = self.us5(predictions, idx5, skip5)
        # dim is (16inputDim)x14x14
        predictions = self.us4(predictions, idx4, skip4)
        # dim is (inputDim8)x28x28
        predictions = self.us3(predictions, idx3, skip3)
        # dim is (4inputDim)x56x56
        predictions = self.us2(predictions, idx2, skip2)
        # dim is (2inputDim)x112x112
        predictions = self.us1(predictions, idx1, skip1)
        # dim is (inputDim)x224x224
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
        self.pool = nn.MaxPool2d(kernel_size=factor, return_indices=True)
        self.relu2 = nn.ReLU()

    def forward(self, inFeatures):
        outFeatures = self.convLayer1(inFeatures)
        outFeatures = self.bn1(outFeatures)
        outFeatures = self.relu1(outFeatures)
        outFeatures = self.convLayer2(outFeatures)
        outFeatures = self.bn2(outFeatures)
        outFeatures = self.relu2(outFeatures)
        outputForSkipConnection = outFeatures
        outFeatures, poolIndices = self.pool(outFeatures)
        return outFeatures, outputForSkipConnection, poolIndices


class UpSamplingBlock(nn.Module):
    def __init__(self, kernelSize, inputFeatures, outputFeatures, skipConnection, factor=2):
        super(UpSamplingBlock, self).__init__()
        self.skipConnection = skipConnection

        self.deconv1 = nn.ConvTranspose2d(in_channels=inputFeatures, out_channels=outputFeatures,
                                          kernel_size=kernelSize, padding=kernelSize // 2)
        self.bn1 = nn.BatchNorm2d(outputFeatures)
        self.relu1 = nn.ReLU()
        self.unpool = nn.MaxUnpool2d(factor)

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

    def forward(self, inFeatures, poolIndices, skipInput):
        outFeatures = self.deconv1(inFeatures)
        outFeatures = self.bn1(outFeatures)
        outFeatures = self.relu1(outFeatures)
        outFeatures = self.unpool(outFeatures, poolIndices)

        if self.skipConnection:
            outFeatures = torch.cat((outFeatures, skipInput), dim=1)
        outFeatures = self.conv2(outFeatures)
        outFeatures = self.bn2(outFeatures)
        outFeatures = self.relu2(outFeatures)

        outFeatures = self.conv3(outFeatures)
        outFeatures = self.bn3(outFeatures)
        outFeatures = self.relu3(outFeatures)
        return outFeatures
