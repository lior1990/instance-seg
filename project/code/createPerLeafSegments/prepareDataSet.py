#   1. load a good checkpoint
#   2. run the model on train2 dataset in the following way:
#       2.1. run every image in the net and get a feature tensor (load the images without augmentations)
#       2.2. for every different label (in the labeled image) do:
#           2.2.1. get the mean in the embedding space of the embeddings that belong to this label
#           2.2.2. create a 224x224 map. each entry will hold a "probability" of belonging to this label's segment per
#                   pixel: 1-distToMean/maxDistToMean. where distToMean is the distance of the embedding to the label mean
#                   and maxDistToMean is the maximum distance (of all embeddings) to the label mean
#           2.2.3. save the label image (224x224) as a binary map and the relevant distance map as torch tensor file

NUM_OF_CHARS = 6

from costum_dataset import CostumeDataset
from utils.objects import DataSetParams
from torch.utils.data import DataLoader
from config import *
import os
import torch
from imageio import imsave
import numpy as np
from utils.objects import DataSetParams


def createOutputLocations(dataOutPath, labelsOutPath):
    os.makedirs(dataOutPath, exist_ok=True)
    os.makedirs(labelsOutPath, exist_ok=True)


def getMeanTensor(features, labels, focusLabel):
    labels = labels.flatten()
    features = features.permute(1, 2, 0).contiguous()
    shape = features.size()
    features = features.view(shape[0] * shape[1], shape[2])
    locations = torch.LongTensor(np.where(labels == focusLabel)[0]).type(long_type)
    # all vectors of this instance
    vectors = torch.index_select(features, dim=0, index=locations).type(double_type)
    meanTensor = vectors.mean(dim=0)
    return meanTensor


def getDistancesMask(features, centerTensor):
    distMap = (features.permute(1, 2, 0).type(double_type) - centerTensor.type(double_type)).norm(dim=2, p=2)
    distMap = distMap / distMap.max()
    distMap = 1 - distMap
    return distMap


def getMask(features, labels, focusLabel):
    meanTensor = getMeanTensor(features, labels, focusLabel)
    distMap = getDistancesMask(features, meanTensor)
    maskVisualization = distMap.cpu().numpy()
    maskVisualization = maskVisualization * 255
    maskVisualization = maskVisualization.astype('uint8')
    maskLabels = np.zeros(labels.shape)
    maskLabels[np.where(labels == focusLabel)] = 255
    maskLabels = maskLabels.astype('uint8')
    return maskLabels, distMap, maskVisualization


def run(dataSetParams, expName, epochName, dataOutPath, labelsOutPath, outIdsFilePath, counter=0):
    createOutputLocations(dataOutPath, labelsOutPath)
    with torch.no_grad():
        dataset = CostumeDataset(dataSetParams.ids_path, dataSetParams.data_folder_path,
                                 dataSetParams.labels_folder_path, mode='val')
        dataloader = DataLoader(dataset)
        logger = config_logger(expName)
        model = getFeatureExtractionModel(name=expName, logger=logger, currentEpoch=epochName)[0]
        model.eval()
        segmentCount = 0
        backgroundCount = 0
        with open(outIdsFilePath, 'w') as file:
            for i, batch in enumerate(dataloader):
                image = batch['image']
                label = batch['label']
                features = model(image, None, None)[0]
                allLabels = np.unique(label)
                for currentLabel in allLabels:
                    if currentLabel == BACKGROUND_LABEL:
                        continue
                    maskLabel, maskDistances, maskVisualization = getMask(features[0], label[0], currentLabel)
                    segmentCount += len(np.where(maskLabel >= 128)[0])
                    backgroundCount += len(np.where(maskLabel < 128)[0])
                    file.write(str(counter).zfill(NUM_OF_CHARS) + '\n')
                    torch.save(maskDistances, dataOutPath + str(counter).zfill(NUM_OF_CHARS) + '.sv')
                    imsave(dataOutPath + str(counter).zfill(NUM_OF_CHARS) + '_visualize.jpg', maskVisualization)
                    imsave(labelsOutPath + str(counter).zfill(NUM_OF_CHARS) + '.png', maskLabel)
                    counter += 1
    return counter, segmentCount, backgroundCount


def main():
    imagesFolderPath = os.path.join('..', '..', '..', 'cvppp', 'formattedA1Only', 'train2', 'images', '')
    labelsFolderPath = os.path.join('..', '..', '..', 'cvppp', 'formattedA1Only', 'train2', 'labels', '')
    imagesIdsFilePath = os.path.join('..', '..', '..', 'cvppp', 'formattedA1Only', 'train2', 'images_ids.txt')

    # dataOutPath = os.path.join('..', '..', '..', 'cvppp', 'formattedA1Only', 'train2', 'mask_distances_no_edges_no_weighted_mean', '')
    dataOutPath = os.path.join('..', '..', '..', 'cvppp', 'formattedA1Only', 'train2', 'mask_distances_no_edges_yes_weighted_mean', '')
    # dataOutPath = os.path.join('..', '..', '..', 'cvppp', 'formattedA1Only', 'train2', 'mask_distances_yes_edges_no_weighted_mean', '')
    # dataOutPath = os.path.join('..', '..', '..', 'cvppp', 'formattedA1Only', 'train2', 'mask_distances_yes_edges_yes_weighted_mean', '')
    # labelsOutPath = os.path.join('..', '..', '..', 'cvppp', 'formattedA1Only', 'train2', 'mask_labels_no_edges_no_weighted_mean', '')
    labelsOutPath = os.path.join('..', '..', '..', 'cvppp', 'formattedA1Only', 'train2', 'mask_labels_no_edges_yes_weighted_mean', '')
    # labelsOutPath = os.path.join('..', '..', '..', 'cvppp', 'formattedA1Only', 'train2', 'mask_labels_yes_edges_no_weighted_mean', '')
    # labelsOutPath = os.path.join('..', '..', '..', 'cvppp', 'formattedA1Only', 'train2', 'mask_labels_yes_edges_yes_weighted_mean', '')
    # outIdsFilePath = os.path.join(os.path.join('..', '..', '..', 'cvppp', 'formattedA1Only', 'train2', 'mask_ids_no_edges_no_weighted_mean.txt'))
    outIdsFilePath = os.path.join(os.path.join('..', '..', '..', 'cvppp', 'formattedA1Only', 'train2', 'mask_ids_no_edges_yes_weighted_mean.txt'))
    # outIdsFilePath = os.path.join(os.path.join('..', '..', '..', 'cvppp', 'formattedA1Only', 'train2', 'mask_ids_yes_edges_no_weighted_mean.txt'))
    # outIdsFilePath = os.path.join(os.path.join('..', '..', '..', 'cvppp', 'formattedA1Only', 'train2', 'mask_ids_yes_edges_yes_weighted_mean.txt'))

    experiment_name = 'best_no_edges_yes_weighted_mean'
    # experiment_name = 'best_no_edges_yes_weighted_mean'
    # experiment_name = 'best_yes_edges_no_weighted_mean'
    # experiment_name = 'best_yes_edges_yes_weighted_mean'
    epochName = '501'
    dataSetParams = DataSetParams(data_folder_path=imagesFolderPath, labels_folder_path=labelsFolderPath,
                                  ids_path=imagesIdsFilePath)
    counter = 0
    totalSegmentCount = 0
    totalBackgroundCount = 0
    for i in range(1):
        print(i)
        counter, segCnt, bgCnt = run(dataSetParams, experiment_name, epochName, dataOutPath, labelsOutPath,
                                     outIdsFilePath, counter)
        totalSegmentCount += segCnt
        totalBackgroundCount += bgCnt
    weight = totalBackgroundCount / totalSegmentCount
    print('the weight of the background over ech segment is', str(weight))


if __name__ == '__main__':
    main()
