import logging
import os
from model import *
import MetricLearningModel

# Hyper parameters
embedding_dim = 64
batch_size = 32

if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    batch_size = batch_size * torch.cuda.device_count()

learning_rate = 0.0003
lr_decay = 0.98
max_epoch_num = 100000
# context = True
context = False

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


def config_experiment(name, resume=True,useBest=False, context=context):

    exp = {}
    try:
        os.makedirs(os.path.join(chkpts_dir, name))
    except:
        pass
    logger = config_logger(name)

    if resume:

        try:
            if useBest:
                exp_path = os.path.join(chkpts_dir,name,'best.pth')
            else:
                exp_path = os.path.join(chkpts_dir, name, "chkpt.pth")
            exp = torch.load(exp_path, map_location=lambda storage, loc: storage)
            logger.info("loading checkpoint, experiment: " + name)
            return exp, logger
        except Exception as e:
            logger.warning('checkpoint does not exist. creating new experiment')

    # fe = FeatureExtractor(embedding_dim, context=context)
    fe = MetricLearningModel.FeatureExtractor(embedding_dim)
    exp['fe_state_dict'] = fe.state_dict()
    exp['epoch'] = 0
    exp['best_dice'] = None
    exp['train_fe_loss'] = []
    exp['val_fe_loss'] = []
    exp['dice_history'] = []

    return exp, logger


def save_experiment(exp, name, isBest=False):
    exp_path = os.path.join(chkpts_dir, name, "chkpt.pth")
    torch.save(exp, exp_path)
    if isBest:
        best_exp_path = os.path.join(chkpts_dir, name, "best.pth")
        torch.save(exp, best_exp_path)


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
