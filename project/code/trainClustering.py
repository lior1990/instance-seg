import torch.autograd
from costum_dataset import *
from torch.utils.data import DataLoader
from evaluate import *
from config import *
import os
import matplotlib
from EmbeddingsClusteringNet import EmbeddingsClustering
from optimizer import *

# matplotlib.use('Agg')
from matplotlib import pyplot as plt


def set_random_seed():
    # set random seed for pseudo-random generators so experiment results can be reproducible
    torch.manual_seed(0)
    np.random.seed(0)


def worker_init_fn(worker_id):
    set_random_seed()


def run(current_experiment, train_data_folder_path, train_labels_folder_path, train_ids_path):
    set_random_seed()
    # Dataloader
    train_dataset = CostumeDataset(train_ids_path, train_data_folder_path, train_labels_folder_path,
                                   mode="train", img_h=224, img_w=224)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4,
                                  worker_init_fn=worker_init_fn)

    # Set up an experiment

    fe, fe_opt, optScheduler, exp_logger, current_epoch, train_fe_loss_history = \
        config_experiment(current_experiment, resume=True, useBest=False)
    fe.eval()

    clusterNet = EmbeddingsClustering(embedding_dim, MAX_NUM_OF_INSTANCES)
    opt = getOptimizer(clusterNet)
    # sched = getStepOptimizerScheduler(opt,-1)
    sched = getCyclicOptimizerScheduler(opt, -1)

    # exp_logger.info('training started/resumed at epoch ' + str(current_epoch))
    exp_logger.info('training started/resumed at epoch ' + str(0))

    # for i in range(current_epoch, trainParams.max_epoch_num):
    for i in range(0, trainParams.max_epoch_num):
        sched.step()
        exp_logger.info('epoch: ' + str(i) + ' LR: ' + str(sched.get_lr()))
        epochAvgLoss = 0
        for batch_num, batch in enumerate(train_dataloader):
            inputs = Variable(batch['image'].type(float_type))
            labels = batch['label'].cpu().numpy()
            labelEdges = batch['labelEdges'].cpu().numpy()
            opt.zero_grad()
            with torch.no_grad():
                features, totLoss, varLoss, distLoss, edgeLoss, regLoss = fe(inputs, labels, labelEdges)
            predictedLabels, totalLoss = clusterNet(features, labels)
            totalLoss = totalLoss.sum()
            totalLoss = totalLoss / batch_size
            totalLoss.backward()
            opt.step()
            epochAvgLoss += totalLoss.cpu().item()

            exp_logger.info('epoch: ' + str(i) + ', batch number: ' + str(batch_num) +
                            ', loss: ' + "{0:.2f}".format(totalLoss))
        epochAvgLoss = epochAvgLoss / (batch_num + 1)
        exp_logger.info('epoch: ' + str(i) + ', average loss: {0:.2f}'.format(epochAvgLoss))

        # if i % trainParams.saveModelIntervalEpochs == 0:
        #     # Save experiment
        #     exp_logger.info('Saving checkpoint...')
        #     save_experiment({'fe_state_dict': fe.state_dict(),
        #                      'epoch': i + 1,
        #                      'train_fe_loss': train_fe_loss_history},
        #                     {'opt_state_dict': fe_opt.state_dict()},
        #                     current_experiment)
        # # Plot and save loss history
        # plt.plot(train_fe_loss_history, 'r')
        # try:
        #     os.makedirs(os.path.join('visualizations', current_experiment))
        # except:
        #     pass
        # plt.savefig(os.path.join('visualizations', current_experiment, 'fe_loss.png'))
        # plt.close()

    return
