import torch.nn as nn
import torch
import config
import numpy as np


class SingleClusterNet(nn.Module):
    def __init__(self, useSkip, useFC, segmentWeight):
        super(SingleClusterNet, self).__init__()
        h = 224
        w = 224
        self.useSkip = useSkip
        self.useFC = useFC
        inputDim = 1
        # dim is (inputDim)x224x224
        ds1OutDim = 2
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

        if useFC:
            fcNumFeatures = ds5OutDim * (h // (2 ** 5)) * (w // (2 ** 5))
            self.fc = nn.Sequential(
                nn.Linear(fcNumFeatures, fcNumFeatures),
                nn.ReLU()
            )

        # dim is (16ds1OutDim)x7x7
        self.us5 = UpSamplingBlock(3, ds5OutDim, ds4OutDim, useSkip)
        # dim is (8ds1OutDim)x14x14
        self.us4 = UpSamplingBlock(3, ds4OutDim, ds3OutDim, useSkip)
        # dim is (4ds1OutDim)x28x28
        self.us3 = UpSamplingBlock(3, ds3OutDim, ds2OutDim, useSkip)
        # dim is (2ds1OutDim)x56x56
        self.us2 = UpSamplingBlock(3, ds2OutDim, ds1OutDim, useSkip)
        # dim is (ds1OutDim)x112x112
        self.us1 = UpSamplingBlock(3, ds1OutDim, inputDim, useSkip)
        # dim is (inputDim)x112x112

        self.lastConv = nn.Conv2d(in_channels=inputDim, out_channels=1, kernel_size=1)
        # dim is 1x224x224
        weight = torch.Tensor([segmentWeight]).type(config.float_type)
        self.loss = nn.BCEWithLogitsLoss(pos_weight=weight)  # reduction is mean

    def forward(self, diffsBatch, labelsBatch):
        batchSize = diffsBatch.shape[0]
        in1 = diffsBatch.type(config.float_type)
        # dim is (inputDim)x224x224
        out1, idx1 = self.ds1(in1)
        # dim is (2inputDim)x112x112
        out2, idx2 = self.ds2(out1)
        # dim is (inputDim4)x56x56
        out3, idx3 = self.ds3(out2)
        # dim is (8inputDim)x28x28
        out4, idx4 = self.ds4(out3)
        # dim is (16inputDim)x14x14
        out5, idx5 = self.ds5(out4)
        # dim is (32inputDim)x7x7
        predictions = out5
        if self.useFC:
            beforeFCShape = out5.shape
            fcIn = out5.view(batchSize, -1)
            fcOut = self.fc(fcIn)
            fcOut = fcOut.view(beforeFCShape)
            predictions = fcOut
        # dim is (32inputDim)x7x7
        predictions = self.us5(predictions, idx5, out4)
        # dim is (16inputDim)x14x14
        predictions = self.us4(predictions, idx4, out3)
        # dim is (inputDim8)x28x28
        predictions = self.us3(predictions, idx3, out2)
        # dim is (4inputDim)x56x56
        predictions = self.us2(predictions, idx2, out1)
        # dim is (2inputDim)x112x112
        predictions = self.us1(predictions, idx1, in1)
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
    def __init__(self, kernelSize, inputFeatures, outputFeatures):
        super(DownSamplingBlock, self).__init__()
        self.convLayer = nn.Conv2d(in_channels=inputFeatures, out_channels=outputFeatures, kernel_size=kernelSize,
                                   padding=kernelSize // 2)
        self.pool = nn.MaxPool2d(kernel_size=2, return_indices=True)
        self.relu = nn.ReLU()

    def forward(self, inFeatures):
        outFeatures = self.convLayer(inFeatures)
        outFeatures, poolIndices = self.pool(outFeatures)
        outFeatures = self.relu(outFeatures)
        return outFeatures, poolIndices


class UpSamplingBlock(nn.Module):
    def __init__(self, kernelSize, inputFeatures, outputFeatures, skipConnection):
        super(UpSamplingBlock, self).__init__()
        self.unpool = nn.MaxUnpool2d(2)
        self.skipConnection = skipConnection
        self.deconv = nn.ConvTranspose2d(in_channels=inputFeatures, out_channels=outputFeatures,
                                         kernel_size=kernelSize, padding=kernelSize // 2)
        convInFeatures = outputFeatures
        if skipConnection:
            convInFeatures *= 2
        self.conv = nn.Conv2d(in_channels=convInFeatures, out_channels=outputFeatures, kernel_size=kernelSize,
                              padding=kernelSize // 2)
        self.relu = nn.ReLU()

    def forward(self, inFeatures, poolIndices, skipInput):
        outFeatures = self.unpool(inFeatures, poolIndices)
        outFeatures = self.deconv(outFeatures)
        if self.skipConnection:
            outFeatures = torch.cat((outFeatures, skipInput), dim=1)
        outFeatures = self.conv(outFeatures)
        outFeatures = self.relu(outFeatures)
        return outFeatures
