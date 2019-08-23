from collections import OrderedDict

from ModelWithLoss import CompleteModel
import config
import torch.nn as nn
from optimizer import *


def loadAll(modelChkpt, optimizerChkpt, lastEpochTrained):
    model = CompleteModel(config.embedding_dim)
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

    if torch.cuda.is_available():
        print("Using CUDA")
        device = torch.device("cuda:0")
        if torch.cuda.device_count() > 1:
            print("Using CUDA with %s GPUs!" % torch.cuda.device_count())
            model = nn.DataParallel(model)
    else:
        device = torch.device("cpu")

    model.to(device)

    optimizer = getOptimizer(model)

    if optimizerChkpt is not None:
        try:
            optimizer.load_state_dict(optimizerChkpt['opt_state_dict'])
        except:
            print('failed to load optimizer state dict!!!!!!!!!!!!!')
            pass

    scheduler = getOptimizerScheduler(optimizer, lastEpochTrained)

    return model, optimizer, scheduler
