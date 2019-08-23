from MetricLearningModel import FeatureExtractor
from lossModule import LossModule
import torch.nn as nn
import config
import torch


class CompleteModel(nn.Module):
    def __init__(self, embeddingDim):
        super(CompleteModel, self).__init__()
        self.fe = FeatureExtractor(embeddingDim)
        self.l = LossModule()
        # self.type(config.double_type)

    def forward(self, imgBatch, lblBatch):
        features = self.fe(imgBatch)
        if self.training:
            totLoss, varLoss, distLoss, edgeLoss, regLoss = self.l(features, lblBatch)
        else:
            totLoss = None
            varLoss = None
            distLoss = None
            edgeLoss = None
            regLoss = None
        return features, totLoss, varLoss, distLoss, edgeLoss, regLoss
