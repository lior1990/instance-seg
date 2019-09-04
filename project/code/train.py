import torch.autograd
from costum_dataset import *
from evaluate import *
from config import *
import os
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt


def set_random_seed():
    # set random seed for pseudo-random generators so experiment results can be reproducible
    torch.manual_seed(0)
    np.random.seed(0)


def worker_init_fn(worker_id):
    set_random_seed()


def run(current_experiment, train_data_set_params, loss_params, sub_experiment_name=''):
    set_random_seed()
    # Dataloader
    train_dataset = CostumeDataset(train_data_set_params.ids_path,
                                   train_data_set_params.data_folder_path,
                                   train_data_set_params.labels_folder_path,
                                   mode="train", img_h=224, img_w=224)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4,
                                  worker_init_fn=worker_init_fn)

    # Set up an experiment
    exp_logger = config_logger(current_experiment, sub_experiment_name)
    model, model_optimizer, optimizer_scheduler, current_epoch, train_loss_history = \
        getFeatureExtractionModel(current_experiment, exp_logger, loss_params=loss_params,
                                  sub_experiment_name=sub_experiment_name, resume=True, useBest=False)

    exp_logger.info('training started/resumed at epoch ' + str(current_epoch))

    for i in range(current_epoch, trainParams.max_epoch_num):
        running_fe_loss = 0
        running_var_loss = 0
        running_dist_loss = 0
        running_edge_loss = 0
        running_reg_loss = 0
        optimizer_scheduler.step()
        exp_logger.info('epoch: ' + str(i) + ' LR: ' + str(optimizer_scheduler.get_lr()))
        for batch_num, batch in enumerate(train_dataloader):
            inputs = Variable(batch['image'].type(float_type))
            labels = batch['label'].cpu().numpy()
            labelEdges = batch['labelEdges'].cpu().numpy()
            model_optimizer.zero_grad()
            features, totLoss, varLoss, distLoss, edgeLoss, regLoss = model(inputs, labels, labelEdges)
            totalLoss = totLoss.sum() / inputs.shape[0]
            varianceLoss = varLoss.sum() / inputs.shape[0]
            distanceLoss = distLoss.sum() / inputs.shape[0]
            edgeToEdgeLoss = edgeLoss.sum() / inputs.shape[0]
            regularizationLoss = regLoss.sum() / inputs.shape[0]
            totalLoss.backward()
            model_optimizer.step()

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

        train_loss = running_fe_loss / (batch_num + 1)
        train_loss_history.append(train_loss)

        if i % trainParams.saveModelIntervalEpochs == 0:
            # Save experiment
            exp_logger.info('Saving checkpoint...')
            save_experiment({'fe_state_dict': model.state_dict(),
                             'epoch': i + 1,
                             'train_fe_loss': train_loss_history},
                            {'opt_state_dict': model_optimizer.state_dict()},
                            current_experiment, sub_experiment_name=sub_experiment_name)
        # Plot and save loss history
        plt.plot(train_loss_history, 'r')
        try:
            os.makedirs(os.path.join('visualizations', current_experiment, sub_experiment_name))
        except:
            pass
        plt.savefig(os.path.join('visualizations', current_experiment, sub_experiment_name, 'fe_loss.png'))
        plt.close()

    # todo: use VAL loss instead of train loss
    return train_loss
