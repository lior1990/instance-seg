from torch.utils.data import Dataset
import config
import PIL.Image as im
import numpy as np
import torch
import os


class SingleClustersDataSet(Dataset):
    def __init__(self, idsFilePath, dataFilePath, labelsFilePath):
        super(SingleClustersDataSet, self).__init__()
        with open(idsFilePath) as idsFile:
            self.ids = idsFile.read().split('\n')[:-1]
        self.length = len(self.ids)
        self.DATA_FILE_TYPE = '.sv'
        self.LABE_FILE_TYPE = '.png'
        self.dataPath = dataFilePath
        self.labelsPath = labelsFilePath

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        id = self.ids[item]
        dataName = os.path.join(self.dataPath, str(id) + self.DATA_FILE_TYPE)
        labelName = os.path.join(self.labelsPath, str(id) + self.LABE_FILE_TYPE)
        data = torch.load(dataName, map_location='cpu')
        if len(data.shape) == 2:
            data = torch.unsqueeze(data, dim=0)
        labelIm = im.open(labelName)
        label = np.asarray(labelIm)
        ZOLabel = np.zeros(label.shape)
        ZOLabel[np.where(label >= 128)] = 1
        if len(ZOLabel.shape) == 2:
            ZOLabel = np.expand_dims(ZOLabel,axis=0)
        return {'data': data, 'label': ZOLabel}
