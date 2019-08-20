from collections import OrderedDict

import torch.autograd
from costum_dataset import *
from torch.utils.data import DataLoader
import torch.nn as nn
from evaluate import *
from config import *
from ModelWithLoss import CompleteModel
import os


def run(current_experiment, train_data_folder_path, train_labels_folder_path, train_ids_path):
    # Dataloader
    train_dataset = CostumeDataset(train_ids_path, train_data_folder_path, train_labels_folder_path,
                                   mode="train", img_h=224, img_w=224)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=2)

    # Set up an experiment
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

    for i in range(current_epoch, trainParams.max_epoch_num):
        adjust_learning_rate(fe_opt, i, trainParams.learning_rate, trainParams.lr_decay)
        running_fe_loss = 0
        for batch_num, batch in enumerate(train_dataloader):
            inputs = Variable(batch['image'].type(float_type))
            labels = batch['label'].cpu().numpy()
            labelEdges = batch['labelEdges'].cpu().numpy()
            fe_opt.zero_grad()
            features, losses = fe(inputs, labels, labelEdges)
            totalLoss = losses.mean()
            totalLoss.backward()
            fe_opt.step()

            np_fe_loss = totalLoss.cpu().item()

            running_fe_loss += np_fe_loss
            exp_logger.info('epoch: ' + str(i) + ', batch number: ' + str(batch_num) +
                            ', loss: ' + "{0:.2f}".format(np_fe_loss))

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


def adjust_learning_rate(optimizer, epoch, lr, decay_rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * np.power(decay_rate, epoch)
