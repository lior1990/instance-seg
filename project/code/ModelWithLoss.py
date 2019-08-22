from MetricLearningModel import FeatureExtractor
from lossModule import LossModule
import torch.nn as nn


class CompleteModel(nn.Module):
    def __init__(self,embeddingDim):
        super(CompleteModel,self).__init__()
        self.fe = FeatureExtractor(embeddingDim)
        self.loss = LossModule()

    def forward(self, imgBatch,lblBatch,lblEdgBatch):
        features = self.fe(imgBatch)
        if self.training:
            totLoss = self.loss(features, lblBatch, lblEdgBatch)
        else:
            totLoss = None
        return features,totLoss