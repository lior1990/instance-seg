from MetricLearningModel import FeatureExtractor
from lossModule import LossModule
import torch.nn as nn
import config


class CompleteModel(nn.Module):
    def __init__(self,embeddingDim):
        super(CompleteModel,self).__init__()
        self.fe = FeatureExtractor(embeddingDim)
        self.l = LossModule()
        self.type(config.double_type)
    def forward(self, imgBatch,lblBatch,lblEdgBatch):
        features = self.fe(imgBatch)
        if self.training:
            totLoss = self.l(features,lblBatch,lblEdgBatch)
        else:
            totLoss = None
        return features,totLoss