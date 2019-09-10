import torch.autograd
from torch.utils.data import DataLoader
from config import *
from ClusterNetDataset import SingleClustersDataSet
from optimizer import *
import numpy as np
from torch.autograd import Variable

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def set_random_seed():
    # set random seed for pseudo-random generators so experiment results can be reproducible
    torch.manual_seed(0)
    np.random.seed(0)


def worker_init_fn(worker_id):
    set_random_seed()


def run(current_experiment, train_data_folder_path, train_labels_folder_path, train_ids_path):
    set_random_seed()
    # Dataloader
    train_dataset = SingleClustersDataSet(train_ids_path, train_data_folder_path, train_labels_folder_path)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=2,
                                  worker_init_fn=worker_init_fn)

    logger = config_logger(current_experiment)
    clusterNet, opt, sched, current_epoch, trainLossHistory = getClusterModel(current_experiment, logger)

    logger.info('cluster learning training started/resumed at epoch ' + str(current_epoch))

    for epochNum in range(current_epoch, trainParams.max_epoch_num):
        sched.step()
        logger.info('cluster learning epoch: ' + str(epochNum) + ' LR: ' + str(sched.get_lr()))
        epochAvgLoss = 0
        totalItems = 0
        for batch_num, batch in enumerate(train_dataloader):
            inputs = Variable(batch['data'].type(float_type))
            labels = batch['label'].cpu().numpy()
            currBatchSize = inputs.shape[0]
            totalItems += currBatchSize
            opt.zero_grad()
            predictedLabels, totalLoss = clusterNet(inputs, labels)
            totalLoss = totalLoss.sum()
            totalLoss = totalLoss / currBatchSize
            totalLoss.backward()
            opt.step()
            epochAvgLoss += totalLoss.cpu().item() * currBatchSize
            logger.info('cluster training epoch: ' + str(epochNum) + ', batch number: ' + str(batch_num) +
                        ', loss: {0:.4f}'.format(totalLoss))
        epochAvgLoss = epochAvgLoss / totalItems
        trainLossHistory.append(epochAvgLoss)
        logger.info('cluster learning epoch: ' + str(epochNum) + ', average loss: {0:.4f}'.format(epochAvgLoss))
        if epochNum % trainParams.saveModelIntervalEpochs == 0:
            # Save experiment
            logger.info('Saving checkpoint...')
            save_experiment({'cl_state_dict': clusterNet.state_dict(),
                             'epoch': epochNum + 1,
                             'train_cl_loss': trainLossHistory},
                            {'opt_state_dict': opt.state_dict()},
                            current_experiment, clusterLearning=True)
        # Plot and save loss history
        plt.plot(trainLossHistory, 'r')
        try:
            os.makedirs(os.path.join('visualizations', current_experiment))
        except:
            pass
        plt.savefig(os.path.join('visualizations', current_experiment, 'clustering_loss.png'))
        plt.close()
    minTrainingLoss = min(trainLossHistory)
    minEpoch = trainLossHistory.index(minTrainingLoss)
    logger.info('Finished cluster learning. Best epoch of training loss is ' + str(minEpoch) +
                ' with loss ' + str(minTrainingLoss))
    return trainLossHistory
