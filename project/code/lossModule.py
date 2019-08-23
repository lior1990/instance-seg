import torch.nn as nn
from loss import calcLoss


class LossModule(nn.Module):
    def __init__(self):
        super(LossModule, self).__init__()

    def forward(self, features, labels):
        return calcLoss(features, labels)
