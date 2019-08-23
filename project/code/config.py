import logging
import os
from utils.model_loader import loadAll
import torch

PIXEL_IGNORE_VAL = 255
PIXEL_BOUNDARY_VAL = 255

# Hyper parameters
embedding_dim = 32
batch_size = 10


class LossParams:
    def __init__(self):
        self.alpha = 1
        self.beta = 1
        self.gamma = 1
        self.delta = 0.001
        self.objectEdgeContributeToLoss = False
        self.edgePixelsMaxNum = 200  # float('inf')
        self.norm = 2
        self.dv = 2
        self.dd = 10


lossParams = LossParams()

if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    batch_size = batch_size * torch.cuda.device_count()


class TrainParams:
    def __init__(self):
        self.maxLR = 1e-4
        self.minLR = 1e-6
        self.momentum = 0.9
        self.useNesterov = True
        self.learning_rate_factor = 0.1
        self.optStepSize = 200
        self.optHalfCycle = 50
        self.max_epoch_num = 1001
        self.saveModelIntervalEpochs = 10


trainParams = TrainParams()

# Checkpoints and logs directory - make sure to set local paths
chkpts_dir = 'model_checkpoints'

if torch.cuda.is_available():
    float_type = torch.cuda.FloatTensor
    double_type = torch.cuda.DoubleTensor
    int_type = torch.cuda.IntTensor
    long_type = torch.cuda.LongTensor
else:
    float_type = torch.FloatTensor
    double_type = torch.DoubleTensor
    int_type = torch.IntTensor
    long_type = torch.LongTensor


def config_experiment(name, resume=True, useBest=False, currentEpoch='latest'):
    exp = None
    optParams = None
    epoch = 0
    lossHistory = []
    try:
        os.makedirs(os.path.join(chkpts_dir, name))
    except:
        pass
    logger = config_logger(name)

    if resume:
        if useBest:
            exp_path = os.path.join(chkpts_dir, name, 'best.pth')
            opt_path = os.path.join(chkpts_dir, name, 'opt_best.pth')
        else:
            exp_path = os.path.join(chkpts_dir, name, 'chkpt_' + currentEpoch + '.pth')
            opt_path = os.path.join(chkpts_dir, name, 'opt_chkpt_' + currentEpoch + '.pth')

        try:
            exp = torch.load(exp_path, map_location=lambda storage, loc: storage)
            logger.info("loading checkpoint, experiment: " + name)
            epoch = exp['epoch']
            lossHistory = exp['train_fe_loss']
        except:
            logger.warning('checkpoint does not exist. creating new experiment')

        try:
            optParams = torch.load(opt_path, map_location=lambda storage, loc: storage)
            logger.info('loading optimizer state, experiment: ' + name)
        except:
            logger.warning('optimizer params for checkpoint does not exist. creating new optimizer')

    model, optimizer, optimizerScheduler = loadAll(exp, optParams,epoch)

    return model, optimizer, optimizerScheduler, logger, epoch, lossHistory


def save_experiment(exp, opt, name, isBest=False):
    exp_path = os.path.join(chkpts_dir, name, "chkpt_" + str(exp['epoch']) + ".pth")
    opt_path = os.path.join(chkpts_dir, name, "opt_chkpt_" + str(exp['epoch']) + ".pth")
    torch.save(exp, exp_path)
    torch.save(opt, opt_path)

    exp_path = os.path.join(chkpts_dir, name, "chkpt_latest.pth")
    opt_path = os.path.join(chkpts_dir, name, "opt_chkpt_latest.pth")
    torch.save(exp, exp_path)
    torch.save(opt, opt_path)

    if isBest:
        best_exp_path = os.path.join(chkpts_dir, name, "best.pth")
        best_opt_path = os.path.join(chkpts_dir, name, "opt_best.pth")
        torch.save(exp, best_exp_path)
        torch.save(opt, best_opt_path)


def config_logger(current_exp):
    logger = logging.getLogger(current_exp)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler2 = logging.FileHandler(os.path.join(chkpts_dir, current_exp, 'log'))
    handler2.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(handler2)

    return logger
