import torch
import config


def getOptimizer(model):
    fe_opt = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config.trainParams.maxLR,
                             momentum=config.trainParams.momentum, nesterov=config.trainParams.useNesterov)
    return fe_opt


def getCyclicOptimizerScheduler(optimizer, lastEpochTrained):
    optScheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=optimizer, base_lr=config.trainParams.minLR,
                                                     max_lr=config.trainParams.maxLR,
                                                     step_size_up=config.trainParams.optHalfCycle,
                                                     last_epoch=lastEpochTrained)
    return optScheduler


def getMultiStepOptimizerScheduler(optimizer, lastEpochTrained):
    optScheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                        milestones=config.trainParams.multiStepEpochs,
                                                        gamma=config.trainParams.learning_rate_factor,
                                                        last_epoch=lastEpochTrained)
    return optScheduler


def getStepOptimizerScheduler(optimizer, lastEpochTrained):
    optScheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=config.trainParams.optStepSize,
                                                   gamma=config.trainParams.learning_rate_factor,
                                                   last_epoch=lastEpochTrained)
    return optScheduler


def getOptimizerScheduler(optimizer, lastEpochTrained):
    if lastEpochTrained == 0:
        lastEpochTrained = -1
    # return getStepOptimizerScheduler(optimizer, lastEpochTrained)
    return getMultiStepOptimizerScheduler(optimizer, lastEpochTrained)
    # return getCyclicOptimizerScheduler(optimizer, lastEpochTrained)
