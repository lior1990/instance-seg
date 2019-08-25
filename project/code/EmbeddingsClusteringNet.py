import torch.nn as nn
import config
import itertools
import torch
from torch.autograd import Variable
import numpy as np

class UpsamplingBlock(nn.Module):
    '''
    For the up-sampling I chose Nearest neighbour over deconvolution, to avoid artifacts in the output.
    If skip is set to True, the input to the forward pass must include skip input - i.e. the equivalent sized output
    of the downsampling backbone (here resnet).

    :param channels_in - number of filters/channels in the input.
    :param channels_in - number of filters/channels in the output.
    :param skip - whether or not to use skip input. recommended to set to true.

    '''
    def __init__(self, channels_in, channels_out, skip=False):
        super(UpsamplingBlock, self).__init__()
        #self.upsamplingLayer = nn.ConvTranspose2d(channels_in, channels_out, kernel_size=2, stride=2)
        self.upsamplingLayer = nn.Sequential(nn.Upsample(scale_factor=2),
                                             nn.Conv2d(channels_in, channels_out, kernel_size=1, stride=1),
                                             nn.ReLU(),
                                             # nn.BatchNorm2d(channels_out)
                                             )

        if skip:
            self.conv1 = nn.Conv2d(2*channels_out, channels_out, kernel_size=3, stride=1, padding=1)
        else:
            self.conv1 = nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1)

        self.convLayer1 = nn.Sequential(self.conv1,
                                        nn.ReLU(),
                                        # nn.BatchNorm2d(channels_out)
                                        )

        self.convLayer2 = nn.Sequential(self.conv2,
                                        nn.ReLU(),
                                        # nn.BatchNorm2d(channels_out)
                                        )

    def forward(self, x, skip_input=None):
        x = self.upsamplingLayer(x)
        if skip_input is not None:
            x = torch.cat((x, skip_input), 1)
        x = self.convLayer1(x)
        x = self.convLayer2(x)
        return x


def getConv(inDim, outDim, ker, mp):
    modules = []
    modules.append(nn.Conv2d(inDim, outDim, ker))
    modules.append(nn.ReLU())
    if mp is not None:
        modules.append(nn.MaxPool2d(mp))
    # modules.append(nn.BatchNorm2d(outDim))
    return nn.Sequential(*modules)


def getUpsample(inDim, outDim, ker):
    return nn.Sequential(nn.Upsample(scale_factor=2),
                         nn.ConvTranspose2d(inDim, outDim, kernel_size=ker, stride=1),
                         nn.ReLU(),
                         nn.BatchNorm2d(outDim))


class SkipExpand(nn.Module):
    def __init__(self, featuresIn, featuresOut, nLevels):
        super(SkipExpand, self).__init__()
        self.featuresIn = featuresIn
        self.featuresOut = featuresOut
        self.modules = []
        for i in range(nLevels - 1):
            self.modules.append(nn.Sequential(
                nn.Conv2d(featuresIn, featuresIn, 1),
                nn.ReLU(),
                nn.BatchNorm2d(featuresIn)
            ))
            featuresIn = 2 * featuresIn
        self.modules.append(nn.Sequential(
            nn.Conv2d(featuresIn, featuresOut, 1),
            nn.ReLU(),
            nn.BatchNorm2d(featuresOut)
        ))

    def forward(self, inFeatures):
        outFeatures = None
        for module in self.modules:
            outFeatures = module(inFeatures)
            inFeatures = torch.cat((inFeatures, outFeatures), dim=1)
        return outFeatures


class EmbeddingsClustering(nn.Module):
    """
    This Module learns a clustering algorithm on the embeddings outputted from the MetricLearningModel
    the input dimension is (b,embedding_dim,h,w), and the output dimension is (b,maxInstances,h,w).
    The output is given as a 1-hot vector per pixel. if the vector of the pixel [x,y] is 1 at index k it means that
    the pixel [x,y] belongs to the k'th instance
    """

    def __init__(self, embeddingDim, maxInstances):
        super(EmbeddingsClustering, self).__init__()
        self.embeddingDim = embeddingDim
        self.maxInstances = maxInstances
        # dim is 32x224x224
        self.layer1 = getConv(embeddingDim, 2 * embeddingDim, 5, 2)
        # dim is 64x110x110
        self.layer2 = getConv(2 * embeddingDim, 2 * embeddingDim, 7, 2)
        # dim is 64x52x52
        self.layer3 = getConv(2 * embeddingDim, 4 * embeddingDim, 5, 2)
        # dim is 128x24x24
        self.layer4 = getConv(4 * embeddingDim, 8 * embeddingDim, 5, 2)
        # dim is 256x10x10

        self.fc = nn.Sequential(
            nn.Linear(256 * 10 * 10, 256 * 10 * 10),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 3),
            # dim is 256x12x12
            nn.ReLU()
        )

        self.layer61 = UpsamplingBlock(256, 128, True)
        # dim is 128x24x24
        self.layer62 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 3),
            nn.ReLU()
        )
        # dim is 128x26x26

        self.layer71 = UpsamplingBlock(128, 64, True)
        # dim is 64x52x52
        self.layer72 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4),
            nn.ReLU()
        )
        # dim is 64x55x55

        self.layer81 = UpsamplingBlock(64, 64, True)
        # dim is 64x110x110
        self.layer82 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3),
            nn.ReLU()
        )
        # dim is 64x112x112

        self.layer9 = UpsamplingBlock(64, maxInstances, False)
        # dim is maxInstancesx224x224


        weight = 1 * torch.ones(maxInstances + 1)
        weight[0] = 0.1
        weight[self.maxInstances] = 0.1
        self.loss = nn.CrossEntropyLoss(ignore_index=config.PIXEL_IGNORE_VAL,
                                        reduction='sum',
                                        weight=weight)  # sum for averaging in the training loop

    def forward(self, featureBatch, labelBatch):
        # dim is 32x224x224
        layer1Out = self.layer1(featureBatch)
        # dim is 64x110x110
        layer2Out = self.layer2(layer1Out)
        # dim is 64x52x52
        layer3Out = self.layer3(layer2Out)
        # dim is 128x24x24
        layer4Out = self.layer4(layer3Out)
        # dim is 256x10x10
        fcIn = torch.flatten(layer4Out, start_dim=1)
        # dim is 25600
        fcOut = self.fc(fcIn)
        # dim is 25600
        fcOut = fcOut.view(featureBatch.shape[0], 256, 10, 10)
        # dim is 256x10x10
        predictions = self.layer5(fcOut)
        # dim is 256x12x12
        predictions = self.layer61(predictions, layer3Out)
        predictions = self.layer62(predictions)
        # dim is 128x26x26
        predictions = self.layer71(predictions, layer2Out)
        predictions = self.layer72(predictions)
        # dim is 64x55x55
        predictions = self.layer81(predictions, layer1Out)
        predictions = self.layer82(predictions)
        # dim is 64x112x112
        predictions = self.layer9(predictions)
        # dim is maxInstancesx224x224

        catDim = list(predictions.size())
        catDim[1] = 1
        predictions = torch.cat((predictions, (predictions.min() - 100) * torch.ones(catDim).type(config.float_type)),
                                dim=1)
        maxVals, predictedLabels = predictions.max(dim=1)  # max on the instances dimension
        loss = None
        if self.training:
            loss = self.getLoss(predictions, predictedLabels, labelBatch)

        return predictedLabels, loss

    def getLoss(self, predictions, predictedLabels, labels):

        batchSize = labels.shape[0]

        matchedLabels = torch.zeros(labels.shape).type(config.long_type)
        for i in range(batchSize):
            label = labels[i, :, :]
            predictedLabel = predictedLabels[i, :, :].cpu().numpy()
            matchedLabel = self.getMatchingLabel(predictedLabel, label)
            matchedLabels[i, :, :] = matchedLabel

        return self.loss(predictions, matchedLabels)

    def getMatchingLabel(self, predictedLabel, label):

        """
            predicted label and label are assumed to be numpy arrays
        """

        uniqueInstances = np.unique(label)
        matchedLabel = config.PIXEL_IGNORE_VAL * np.ones(label.shape)
        foundInstances = []

        for instance in uniqueInstances:

            # all places in the label of this instance
            locations = np.where(label == instance)

            intersectedInstances, counts = np.unique(predictedLabel[locations], return_counts=True)
            # find all instance matches for the current instance mask
            topMatches = intersectedInstances[counts.argsort()[::-1]]
            matchedInstance = config.MAX_NUM_OF_INSTANCES  # undefined instance
            for match in topMatches:
                if match not in foundInstances:
                    matchedInstance = match
                    break
            # matchedInstance is the most intersected instance we didnt already find
            matchedLabel[locations] = matchedInstance  # paint instance with the new label
            foundInstances.append(matchedInstance)  # make sure that the same label won't be chosen twice

        # bgLoc = np.where(uniqueInstances == config.BACKGROUND_LABEL)
        # bgLoc = bgLoc[0][0]
        # matchedBg = foundInstances[bgLoc]
        # matchedLabel[np.where(matchedLabel == config.PIXEL_IGNORE_VAL)] = matchedBg

        return torch.from_numpy(matchedLabel).type(config.long_type)
