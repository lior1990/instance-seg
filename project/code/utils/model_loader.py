from collections import OrderedDict

from ModelWithLoss import CompleteModel
from config import embedding_dim
from optimizer import *


def loadAll(modelChkpt, optimizerChkpt, optimizerSchedulerChkpt):
    model = CompleteModel(embedding_dim)
    optimizer = getOptimizer(model)
    scheduler = getOptimizerScheduler(optimizer)

    if modelChkpt is not None:
        try:
            model.load_state_dict(modelChkpt['fe_state_dict'])
        except:
            state_dict = OrderedDict()
            prefix = 'module.'
            for key, val in modelChkpt['fe_state_dict'].items():
                if key.startswith(prefix):
                    key = key[len(prefix):]
                state_dict[key] = val
            model.load_state_dict(state_dict)
        optimizer = getOptimizer(model)

    if optimizerChkpt is not None:
        try:
            optimizer.load_state_dict(optimizerChkpt['opt_state_dict'])
        except:
            print('failed to load optimizer state dict!!!!!!!!!!!!!')
            pass
        scheduler = getOptimizerScheduler(optimizer)

    if optimizerSchedulerChkpt is not None:
        try:
            scheduler.load_state_dict(optimizerSchedulerChkpt['sched_state_dict'])
        except:
            print('failed to load optimizer scheduler state dict!!!!!!!!!!!!!')
            pass

    return model, optimizer, scheduler
