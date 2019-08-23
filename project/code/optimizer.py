import torch
from config import trainParams


def getOptimizer(model):
    fe_opt = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=trainParams.maxLR,
                             momentum=trainParams.momentum, nesterov=trainParams.useNesterov)
    return fe_opt


def getCyclicOptimizerScheduler(optimizer):
    optScheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=optimizer, base_lr=trainParams.minLR,
                                                     max_lr=trainParams.maxLR, step_size_up=trainParams.optHalfCycle)
    return optScheduler


def getStepOptimizerScheduler(optimizer):
    optScheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=trainParams.optStepSize,
                                                   gamma=trainParams.learning_rate_factor)
    return optScheduler

def getOptimizerScheduler(optimizer):
    return getCyclicOptimizerScheduler(optimizer)
