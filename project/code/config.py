import logging
import os
import MetricLearningModel
import torch

# Hyper parameters
embedding_dim = 32
batch_size = 64
class LossParams:
    def __init__(self):
        self.alpha = 1
        self.beta = 1
        self.gamma = 0.001
        self.norm = 2
        self.dv = 0.5
        self.dd = 1.5
lossParams = LossParams()

if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    batch_size = batch_size * torch.cuda.device_count()

learning_rate = 0.0003
lr_decay = 0.98
max_epoch_num = 100000

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


def config_experiment(name, resume=True, useBest=False):
    exp = {}
    try:
        os.makedirs(os.path.join(chkpts_dir, name))
    except:
        pass
    logger = config_logger(name)

    if resume:

        try:
            if useBest:
                exp_path = os.path.join(chkpts_dir, name, 'best.pth')
            else:
                exp_path = os.path.join(chkpts_dir, name, "latest.pth")
            exp = torch.load(exp_path, map_location=lambda storage, loc: storage)
            logger.info("loading checkpoint, experiment: " + name)
            return exp, logger
        except Exception as e:
            logger.warning('checkpoint does not exist. creating new experiment')

    # fe = FeatureExtractor(embedding_dim, context=context)
    fe = MetricLearningModel.FeatureExtractor(embedding_dim)
    exp['epoch'] = 0
    exp['fe_state_dict'] = fe.state_dict()
    exp['train_fe_loss'] = []
    # exp['best_dice'] = None
    # exp['val_fe_loss'] = []
    # exp['dice_history'] = []

    return exp, logger


def save_experiment(exp, name, isBest=False):
    exp_path = os.path.join(chkpts_dir, name, "chkpt_"+str(exp['epoch'])+".pth")
    torch.save(exp,exp_path)
    exp_path = os.path.join(chkpts_dir, name, "latest.pth")
    torch.save(exp,exp_path)
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
