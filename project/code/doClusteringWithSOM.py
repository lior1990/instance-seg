import os

os.environ["CUDA_VISIBLE_DEVICES"] = '4'
from torch.utils.data import DataLoader
from costum_dataset import CostumeDataset
from config import *
import torch
from som import ParallelBatchSOM
from torch.autograd import Variable
import numpy as np
from scipy.misc import imsave


with torch.no_grad():
    # defaultDataPath = os.path.join('..', '..', 'cvppp', 'formatted', 'train2', 'images', '')
    defaultDataPath = os.path.join('..', '..', 'cvppp', 'all_plants', 'train2', 'images', '')
    # defaultLabelsPath = os.path.join('..', '..', 'cvppp', 'formatted', 'train2', 'labels', '')
    defaultLabelsPath = os.path.join('..', '..', 'cvppp', 'all_plants', 'train2', 'labels', '')
    # defaultIdsFile = os.path.join('..', '..', 'cvppp', 'formatted', 'train2', 'images_ids.txt')
    defaultIdsFile = os.path.join('..', '..', 'cvppp', 'all_plants', 'train2', 'images_ids.txt')
    saveFileLoc = os.path.join('.', 'SOM_out')
    currentExperiment = 'leafs_cluster_net_test'
    currentEpoch = 'latest'

    dataset = CostumeDataset(defaultIdsFile, defaultDataPath, defaultLabelsPath, img_h=224, img_w=224)
    dataloader = DataLoader(dataset)

    # Set up an experiment
    fe, optimizer, optimizerScheduler, logger, epoch, lossHistory = \
        config_experiment(currentExperiment, resume=True, useBest=False, currentEpoch=currentEpoch)
    fe.eval()
    for i, batch in enumerate(dataloader):
        try:
            inputs = Variable(batch['image'].type(float_type))
            labels = batch['label'].cpu().numpy()
            features, totLoss, varLoss, distLoss, edgeLoss, regLoss = fe(inputs, None, None)
            if len(features.shape) == 3:
                features = features.repeat(1, 1, 1, 1)
            features = features.flatten(start_dim=2)  # look at the pixels as a vector instead of matrix
            features = features.permute(0, 2, 1)  # rows=#of pixels, cols=embeddingDim
            som = ParallelBatchSOM(10, 10, 32)
            som.initialize('2dgrid')
            sigma0 = 4
            for j in range(1):
                sigma = sigma0 * np.exp(-j / 200)
                print(som.update(features, sigma, True))
            predictions = som.find_bmu(features)
            predictions = predictions.view(224, -1).cpu().numpy()
            savePath = os.path.join(saveFileLoc, 'image' + str(i) + '.jpg')
            imsave(savePath, predictions)


        except:
            print('got exception')
