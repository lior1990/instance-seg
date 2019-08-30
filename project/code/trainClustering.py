import torch.autograd
from torch.utils.data import DataLoader
from config import *
from ClusterNet import SingleClusterNet
from ClusterNetDataset import SingleClustersDataSet
from optimizer import *
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
# matplotlib.use('Agg')


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
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4,
                                  worker_init_fn=worker_init_fn)

    backgroundCount = 0
    segmentCount = 0
    for i, batch in enumerate(train_dataloader):
        labels = batch['label'].cpu().numpy()
        for l in labels:
            ll = l[0]
            backgroundCount = backgroundCount + len(np.where(ll < 0.5)[0])
            segmentCount = segmentCount + len(np.where(ll > 0.5)[0])
    weight = backgroundCount / segmentCount

    # clusterNet = EmbeddingsClustering(embedding_dim, MAX_NUM_OF_INSTANCES)
    clusterNet = SingleClusterNet(useSkip=True, useFC=True, segmentWeight=weight)
    if torch.cuda.is_available():
        print("Using CUDA")
        device = torch.device("cuda:0")
        if torch.cuda.device_count() > 1:
            print("Using CUDA with %s GPUs!" % torch.cuda.device_count())
            clusterNet = nn.DataParallel(clusterNet)
    else:
        device = torch.device("cpu")

    clusterNet.to(device)
    opt = getOptimizer(clusterNet)
    current_epoch = 0
    # sched = getOptimizerScheduler(opt, current_epoch)
    sched = getStepOptimizerScheduler(opt, -1)
    # # exp_logger.info('training started/resumed at epoch ' + str(current_epoch))
    # exp_logger.info('training started/resumed at epoch ' + str(0))
    #
    for i in range(current_epoch, trainParams.max_epoch_num):
        sched.step()
        print('epoch: ' + str(i) + ' learning rate: ' + str(sched.get_lr()))
        #     exp_logger.info('epoch: ' + str(i) + ' LR: ' + str(sched.get_lr()))
        #     epochAvgLoss = 0
        for batch_num, batch in enumerate(train_dataloader):
            inputs = Variable(batch['data'].type(float_type))
            labels = batch['label'].cpu().numpy()
            currBatchSize = inputs.shape[0]
            opt.zero_grad()
            # with torch.no_grad():
            #     features, totLoss, varLoss, distLoss, edgeLoss, regLoss = fe(inputs, labels, labelEdges)
            predictedLabels, totalLoss = clusterNet(inputs, labels)
            totalLoss = totalLoss.sum()
            totalLoss = totalLoss / currBatchSize
            totalLoss.backward()
            opt.step()
            # epochAvgLoss += totalLoss.cpu().item()
            print('epoch: ' + str(i) + ' batch: ' + str(batch_num) + ' loss: {0:.4f}'.format(totalLoss.cpu().item()))
    #
    #         exp_logger.info('epoch: ' + str(i) + ', batch number: ' + str(batch_num) +
    #                         ', loss: ' + "{0:.2f}".format(totalLoss))
    #     epochAvgLoss = epochAvgLoss / (batch_num + 1)
    #     exp_logger.info('epoch: ' + str(i) + ', average loss: {0:.2f}'.format(epochAvgLoss))
    #
    #     # if i % trainParams.saveModelIntervalEpochs == 0:
    #     #     # Save experiment
    #     #     exp_logger.info('Saving checkpoint...')
    #     #     save_experiment({'fe_state_dict': fe.state_dict(),
    #     #                      'epoch': i + 1,
    #     #                      'train_fe_loss': train_fe_loss_history},
    #     #                     {'opt_state_dict': fe_opt.state_dict()},
    #     #                     current_experiment)
    #     # # Plot and save loss history
    #     # plt.plot(train_fe_loss_history, 'r')
    #     # try:
    #     #     os.makedirs(os.path.join('visualizations', current_experiment))
    #     # except:
    #     #     pass
    #     # plt.savefig(os.path.join('visualizations', current_experiment, 'fe_loss.png'))
    #     # plt.close()

    return
