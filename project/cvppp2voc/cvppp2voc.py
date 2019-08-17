from os import listdir
from os import makedirs
from random import sample
import PIL.Image as im
from imageio import imsave
import matplotlib.pyplot as plt
import numpy as np

PREFIX = 'plant'
LABEL_SUFFIX = '_label'
IMAGE_SUFFIX = '_rgb'
ORIGIN_IMG_TYPE = '.png'
ORIGIN_LBL_TYPE = '.png'
DST_IMG_TYPE = '.jpg'
DST_LBL_TYPE = '.png'
IMAGES_FOLDER_NAME = 'images'
LABELS_FOLDER_NAME = 'labels'


def getId(name, pre, suf):
    id = name[len(pre):]
    id = id[:len(id) - len(suf)]
    return id


def getImId(name):
    return getId(name, PREFIX, IMAGE_SUFFIX + ORIGIN_IMG_TYPE)


def getLblId(name):
    return getId(name, PREFIX, LABEL_SUFFIX + ORIGIN_LBL_TYPE)


def convert(sourceFolderPath, destFolderPath, numOfValImages):
    allFiles = listdir(sourceFolderPath)
    images = []
    labels = []
    for file in allFiles:
        if file.startswith(PREFIX) and file.endswith(IMAGE_SUFFIX + ORIGIN_IMG_TYPE):
            images.append(file)
        elif file.startswith(PREFIX) and file.endswith(LABEL_SUFFIX + ORIGIN_LBL_TYPE):
            labels.append(file)

    images.sort()
    labels.sort()
    assert (len(images) == len(labels))
    allIndexes = range(len(images))
    valIndexes = sorted(sample(allIndexes, numOfValImages))
    trainIndexes = list(allIndexes)
    for valIndex in valIndexes:
        trainIndexes.remove(valIndex)

    trainDataPath = destFolderPath + 'train\\'
    valDataPath = destFolderPath + 'val\\'
    try:
        makedirs(trainDataPath)
    except:
        pass

    try:
        makedirs(valDataPath)
    except:
        pass

    try:
        makedirs(trainDataPath + '\\' + IMAGES_FOLDER_NAME + '\\')
    except:
        pass

    try:
        makedirs(trainDataPath + '\\' + LABELS_FOLDER_NAME + '\\')
    except:
        pass

    try:
        makedirs(valDataPath + '\\' + IMAGES_FOLDER_NAME + '\\')
    except:
        pass

    try:
        makedirs(valDataPath + '\\' + LABELS_FOLDER_NAME + '\\')
    except:
        pass

    trainDataFilePath = trainDataPath + 'images_ids.txt'
    with open(trainDataFilePath,'w') as trainIdsFile:
        create(images, labels, sourceFolderPath, trainDataPath, trainIdsFile, trainIndexes)

    valDataFilePath = valDataPath + 'images_ids.txt'
    with open(valDataFilePath,'w') as valIdsFile:
        create(images, labels, sourceFolderPath, valDataPath, valIdsFile, valIndexes)



def create(images, labels, sourceFolderPath, savePath, idFile, indexes):
    for file in indexes:
        imgName = images[file]
        lblName = labels[file]
        img = np.array(im.open(sourceFolderPath + imgName))
        if img.shape[2] == 4:
            img = np.delete(img, 3, 2)

        lbl = np.array(im.open(sourceFolderPath + lblName))

        saveIdIm = getImId(imgName)
        saveIdLbl = getLblId(lblName)
        assert (saveIdIm == saveIdLbl)
        idFile.write(saveIdIm + '\n')
        imsave(savePath + '\\images\\' + str(saveIdIm) + DST_IMG_TYPE, img, DST_IMG_TYPE)
        imsave(savePath + '\\labels\\' + str(saveIdLbl) + DST_LBL_TYPE, lbl, DST_LBL_TYPE)


if __name__ == '__main__':
    convert('C:\\Git\\instance-seg\\cvppp\\training\\A1\\', 'C:\\Git\\instance-seg\\cvppp\\formatted\\', 20)
