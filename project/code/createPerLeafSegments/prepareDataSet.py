#   todo: need to implement a conversion of the image+labels to a set of clusters
#   the following has to be implemented:
#   1. load a good checkpoint (TBD)
#   2. run the model on train2 dataset in the following way:
#       2.1. run every image in the net and get a feature tensor (load the images without augmentations)
#       2.2. for every different label (in the labeled image) do:
#           2.2.1. get the mean in the embedding space of the embeddings that belong to this label
#           2.2.2. create a 224x224 map. each entry will hold a "probability" of belonging to this label's segment per
#                   pixel: 1-distToMean/maxDistToMean. where distToMean is the distance of the embedding to the label mean
#                   and maxDistToMean is the maximum distance (of all embeddings) to the label mean
#           2.2.3. save the label image (224x224) as a binary map and the relevant distance map as torch tensor file

from costum_dataset import CostumeDataset
from utils.objects import DataSetParams
from torch.utils.data import DataLoader
from config import *
import os
import torch
import numpy as np
from utils.objects import DataSetParams

def createOutputLocations(dataOutPath,labelsOutPath,outIdsFilePath):
    os.makedirs(dataOutPath,exist_ok=True)
    os.makedirs(labelsOutPath,exist_ok=True)

def run(dataSetParams,expName,epochName,dataOutPath,labelsOutPath,outIdsFilePath):
    with torch.no_grad():
        dataset = CostumeDataset(dataSetParams['ids_path'],dataSetParams['data_folder_path'],dataSetParams['labels_folder_path'])
        dataloader = DataLoader(dataset)
        model = config_experiment(name=expName,currentEpoch=epochName)[0]
        model.eval()

        for i,batch in enumerate(dataloader):
            image = batch['image']
            label = batch['label']
            features = model(image,None,None)[0]
            allLabels = np.unique(label)
            for currentLabel in allLabels:
                if currentLabel == BACKGROUND_LABEL:
                    continue






def main():
    imagesFolderPath = os.path.join()
    labelsFolderPath = os.path.join()
    imagesIdsFilePath = os.path.join()
    dataOutPath = os.path.join()
    labelsOutPath = os.path.join()
    outIdsFilePath = os.path.join()
    experiment_name = ''
    epochName = ''
    dataSetParams = DataSetParams(data_folder_path=imagesFolderPath,labelsFolderPath=labelsFolderPath,ids_path=imagesIdsFilePath)
    run(dataSetParams,experiment_name,epochName,dataOutPath,labelsOutPath,outIdsFilePath)


if __name__=='__main__':
    main()