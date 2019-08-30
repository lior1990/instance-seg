import logging
import os
from utils.model_loader import loadAll
import torch

PIXEL_BOUNDARY_VAL = 255
PIXEL_IGNORE_VAL = PIXEL_BOUNDARY_VAL
BACKGROUND_LABEL = 0

# Hyper parameters
embedding_dim = 32
batch_size = 5
# MAX_NUM_OF_INSTANCES = 50


if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    batch_size = batch_size * torch.cuda.device_count()


class TrainParams:
    def __init__(self):
        self.maxLR = 0.01  # 1e-4
        self.minLR = 0.001  # 1e-6
        self.momentum = 0.9
        self.useNesterov = True
        self.learning_rate_factor = 0.5
        self.optStepSize = 100
        self.optHalfCycle = 50
        self.max_epoch_num = 10001
        self.saveModelIntervalEpochs = 50


trainParams = TrainParams()

# Checkpoints and logs directory - make sure to set local paths
feature_extraction_chkpts_dir = 'model_checkpoints'
cluster_chkpts_dir = 'cluster_model_checkpoints'
logsDir = 'logs'

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


def getFeatureExtractionModel(name, logger, loss_params=None, sub_experiment_name='', resume=True, useBest=False,
                              currentEpoch='latest'):
    exp = None
    optParams = None
    epoch = 0
    lossHistory = []
    try:
        os.makedirs(os.path.join(feature_extraction_chkpts_dir, name, sub_experiment_name))
    except:
        pass

    if resume:
        if useBest:
            exp_path = os.path.join(feature_extraction_chkpts_dir, name, sub_experiment_name, 'best.pth')
            opt_path = os.path.join(feature_extraction_chkpts_dir, name, sub_experiment_name, 'opt_best.pth')
        else:
            exp_path = os.path.join(feature_extraction_chkpts_dir, name, sub_experiment_name,
                                    'chkpt_' + currentEpoch + '.pth')
            opt_path = os.path.join(feature_extraction_chkpts_dir, name, sub_experiment_name,
                                    'opt_chkpt_' + currentEpoch + '.pth')

        try:
            exp = torch.load(exp_path, map_location=lambda storage, loc: storage)
            logger.info("loading checkpoint, experiment: %s, sub exp: %s", name, sub_experiment_name)
            epoch = exp['epoch']
            lossHistory = exp['train_fe_loss']
        except:
            logger.warning('checkpoint does not exist. creating new experiment')

        try:
            optParams = torch.load(opt_path, map_location=lambda storage, loc: storage)
            logger.info('loading optimizer state, experiment: ' + name)
        except:
            logger.warning('optimizer params for checkpoint does not exist. creating new optimizer')

    model, optimizer, optimizerScheduler = loadAll(exp, optParams, epoch, loss_params, False)

    return model, optimizer, optimizerScheduler, epoch, lossHistory


def getClusterModel(name, logger, sub_experiment_name='', resume=True, useBest=False, currentEpoch='latest'):
    exp = None
    optParams = None
    epoch = 0
    lossHistory = []
    try:
        os.makedirs(os.path.join(cluster_chkpts_dir, name, sub_experiment_name))
    except:
        pass

    if resume:
        if useBest:
            exp_path = os.path.join(cluster_chkpts_dir, name, sub_experiment_name, 'best.pth')
            opt_path = os.path.join(cluster_chkpts_dir, name, sub_experiment_name, 'opt_best.pth')
        else:
            exp_path = os.path.join(cluster_chkpts_dir, name, sub_experiment_name,
                                    'chkpt_' + currentEpoch + '.pth')
            opt_path = os.path.join(cluster_chkpts_dir, name, sub_experiment_name,
                                    'opt_chkpt_' + currentEpoch + '.pth')

        try:
            exp = torch.load(exp_path, map_location=lambda storage, loc: storage)
            logger.info("loading checkpoint, experiment: %s, sub exp: %s", name, sub_experiment_name)
            epoch = exp['epoch']
            lossHistory = exp['train_cl_loss']
        except:
            logger.warning('checkpoint does not exist. creating new experiment')

        try:
            optParams = torch.load(opt_path, map_location=lambda storage, loc: storage)
            logger.info('loading optimizer state, experiment: ' + name)
        except:
            logger.warning('optimizer params for checkpoint does not exist. creating new optimizer')

    model, optimizer, optimizerScheduler = loadAll(exp, optParams, epoch, None, True)

    return model, optimizer, optimizerScheduler, epoch, lossHistory


def save_experiment(exp, opt, name, sub_experiment_name='', isBest=False, clusterLearning=False):
    chkpts_dir = cluster_chkpts_dir if clusterLearning else feature_extraction_chkpts_dir
    try:
        os.makedirs(os.path.join(chkpts_dir, name, sub_experiment_name))
    except:
        pass
    exp_path = os.path.join(chkpts_dir, name, sub_experiment_name, "chkpt_" + str(exp['epoch']) + ".pth")
    opt_path = os.path.join(chkpts_dir, name, sub_experiment_name, "opt_chkpt_" + str(exp['epoch']) + ".pth")
    torch.save(exp, exp_path)
    torch.save(opt, opt_path)

    exp_path = os.path.join(chkpts_dir, name, sub_experiment_name, "chkpt_latest.pth")
    opt_path = os.path.join(chkpts_dir, name, sub_experiment_name, "opt_chkpt_latest.pth")
    torch.save(exp, exp_path)
    torch.save(opt, opt_path)

    if isBest:
        best_exp_path = os.path.join(chkpts_dir, name, sub_experiment_name, "best.pth")
        best_opt_path = os.path.join(chkpts_dir, name, sub_experiment_name, "opt_best.pth")
        torch.save(exp, best_exp_path)
        torch.save(opt, best_opt_path)


def config_logger(current_exp, sub_experiment_name=""):
    logger = logging.getLogger(sub_experiment_name) if sub_experiment_name else logging.getLogger(current_exp)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler2 = logging.FileHandler(os.path.join(logsDir, current_exp, sub_experiment_name, 'log'))
    handler2.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(handler2)

    return logger
