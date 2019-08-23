from collections import OrderedDict

import torch.autograd
from costum_dataset import *
from torch.utils.data import DataLoader
import torch.nn as nn
from evaluate import *
from config import *
from ModelWithLoss import CompleteModel
import os

from utils.model_loader import load_model_from_experiment


def set_random_seed():
    # set random seed for pseudo-random generators so experiment results can be reproducible
    torch.manual_seed(0)
    np.random.seed(0)


def worker_init_fn(worker_id):
    set_random_seed()


set_random_seed()


def run(current_experiment, train_data_folder_path, train_labels_folder_path, train_ids_path):
    # Dataloader
    train_dataset = CostumeDataset(train_ids_path, train_data_folder_path, train_labels_folder_path,
                                   mode="train", img_h=224, img_w=224)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=8,
                                  worker_init_fn=worker_init_fn)

    # Set up an experiment
    experiment, exp_logger = config_experiment(current_experiment, resume=True, useBest=False)

    fe = load_model_from_experiment(experiment)
    experiment, optParams, exp_logger = config_experiment(current_experiment, resume=True, useBest=False)

    fe = CompleteModel(embedding_dim)

    try:
        fe.load_state_dict(experiment['fe_state_dict'])
    except:
        state_dict = OrderedDict()
        prefix = 'module.'
        for key, val in experiment['fe_state_dict'].items():
            if key.startswith(prefix):
                key = key[len(prefix):]
            state_dict[key] = val
        fe.load_state_dict(state_dict)
    current_epoch = experiment['epoch']
    train_fe_loss_history = experiment['train_fe_loss']

    exp_logger.info('training started/resumed at epoch ' + str(current_epoch))

    if torch.cuda.is_available():
        print("Using CUDA")
        device = torch.device("cuda:0")
        if torch.cuda.device_count() > 1:
            print("Using CUDA with %s GPUs!" % torch.cuda.device_count())
            fe = nn.DataParallel(fe)
    else:
        device = torch.device("cpu")

    fe.to(device)

    fe_opt = torch.optim.SGD(filter(lambda p: p.requires_grad, fe.parameters()), lr=trainParams.learning_rate,
                             momentum=0.9, nesterov=True)
    try:
        fe_opt.load_state_dict(optParams['opt_state_dict'])
        print('loaded optimizer state')
    except:
        print('couldnt load optimizer state dict, using defaults')
        fe_opt = torch.optim.SGD(filter(lambda p: p.requires_grad, fe.parameters()), lr=trainParams.learning_rate,
                                 momentum=0.9, nesterov=True)


    # optScheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=fe_opt,factor=trainParams.learning_rate_factor
    #                                                           ,patience=trainParams.learning_rate_patience,
    #                                                           threshold=trainParams.lossPlateuThreshold,
    #                                                           verbose=True)
    # optScheduler = torch.optim.lr_scheduler.StepLR(optimizer=fe_opt,step_size=200,gamma=0.1,last_epoch=-1)
    optScheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=fe_opt,base_lr=1e-6,max_lr=1e-4,step_size_up=50)
    for i in range(current_epoch, trainParams.max_epoch_num):
        running_fe_loss = 0
        running_var_loss = 0
        running_dist_loss = 0
        running_edge_loss = 0
        running_reg_loss = 0
        optScheduler.step()
        exp_logger.info('epoch: ' + str(i) + ' LR: ' + str(optScheduler.get_lr()))
        for batch_num, batch in enumerate(train_dataloader):
            inputs = Variable(batch['image'].type(float_type))
            labels = batch['label'].cpu().numpy()
            labelEdges = batch['labelEdges'].cpu().numpy()
            fe_opt.zero_grad()
            features, totLoss, varLoss, distLoss, edgeLoss, regLoss = fe(inputs, labels, labelEdges)
            totalLoss = totLoss.sum() / batch_size
            varianceLoss = varLoss.sum() / batch_size
            distanceLoss = distLoss.sum() / batch_size
            edgeToEdgeLoss = edgeLoss.sum() / batch_size
            regularizationLoss = regLoss.sum() / batch_size
            totalLoss.backward()
            fe_opt.step()

            np_fe_loss = totalLoss.cpu().item()
            np_var_loss = varianceLoss.cpu().item()
            np_dist_loss = distanceLoss.cpu().item()
            np_edge_loss = edgeToEdgeLoss.cpu().item()
            np_reg_loss = regularizationLoss.cpu().item()

            running_fe_loss += np_fe_loss
            running_var_loss += np_var_loss
            running_dist_loss += np_dist_loss
            running_edge_loss += np_edge_loss
            running_reg_loss += np_reg_loss
            exp_logger.info('epoch: ' + str(i) + ', batch number: ' + str(batch_num) +
                            ', loss: ' + "{0:.2f}".format(np_fe_loss) +
                            ', var loss: ' + '{0:.2f}'.format(np_var_loss) +
                            ', dist loss: ' + '{0:.2f}'.format(np_dist_loss) +
                            ', edge loss: ' + '{0:.2f}'.format(np_edge_loss) +
                            ', reg loss: ' + '{0:.2f}'.format(np_reg_loss))

        train_fe_loss = running_fe_loss / (batch_num + 1)
        train_fe_loss_history.append(train_fe_loss)

        if i % trainParams.saveModelIntervalEpochs == 0:
            # Save experiment
            exp_logger.info('Saving checkpoint...')
            save_experiment({'fe_state_dict': fe.state_dict(),
                             'epoch': i + 1,
                             'train_fe_loss': train_fe_loss_history},
                            {'opt_state_dict': fe_opt.state_dict()},
                            current_experiment)
        # Plot and save loss history
        plt.plot(train_fe_loss_history, 'r')
        try:
            os.makedirs(os.path.join('visualizations', current_experiment))
        except:
            pass
        plt.savefig(os.path.join('visualizations', current_experiment, 'fe_loss.png'))
        plt.close()

    return
