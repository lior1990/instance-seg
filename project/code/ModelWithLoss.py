from MetricLearningModel import FeatureExtractor
from loss import LossModule
import torch.nn as nn


class CompleteModel(nn.Module):
    def __init__(self, embeddingDim, loss_params=None):
        super(CompleteModel, self).__init__()
        self.fe = FeatureExtractor(embeddingDim)
        self.loss = LossModule(loss_params)

    def forward(self, imgBatch, lblBatch, lblEdgBatch):
        features = self.fe(imgBatch)
        if self.training:
            totLoss, varLoss, distLoss, edgeLoss, regLoss = self.loss(features, lblBatch, lblEdgBatch)
        else:
            totLoss = None
            varLoss = None
            distLoss = None
            edgeLoss = None
            regLoss = None
        return features, totLoss, varLoss, distLoss, edgeLoss, regLoss
