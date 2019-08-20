import torch.nn as nn
from loss import calcLoss


class LossModule(nn.Module):
    def __init__(self):
        super(LossModule, self).__init__()


    def forward(self, features, labels, labelEdges):
        return calcLoss(features, labels, labelEdges)
