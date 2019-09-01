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


def convert(sourceFolderPath, destFolderPath, numOfTrain1Images, numOfValImages):
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
    train2Indexes = list(allIndexes)
    for valIndex in valIndexes:
        train2Indexes.remove(valIndex)
    train1Indexes = sorted(sample(train2Indexes, numOfTrain1Images))
    for train1Idx in train1Indexes:
        train2Indexes.remove(train1Idx)

    train1DataPath = destFolderPath + 'train1\\'
    train2DataPath = destFolderPath + 'train2\\'
    valDataPath = destFolderPath + 'val\\'
    try:
        makedirs(train1DataPath)
    except:
        pass
    try:
        makedirs(train2DataPath)
    except:
        pass

    try:
        makedirs(valDataPath)
    except:
        pass

    try:
        makedirs(train1DataPath + '\\' + IMAGES_FOLDER_NAME + '\\')
    except:
        pass
    try:
        makedirs(train2DataPath + '\\' + IMAGES_FOLDER_NAME + '\\')
    except:
        pass

    try:
        makedirs(train1DataPath + '\\' + LABELS_FOLDER_NAME + '\\')
    except:
        pass
    try:
        makedirs(train2DataPath + '\\' + LABELS_FOLDER_NAME + '\\')
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

    train1DataFilePath = train1DataPath + 'images_ids.txt'
    with open(train1DataFilePath, 'w') as train1IdsFile:
        create(images, labels, sourceFolderPath, train1DataPath, train1IdsFile, train1Indexes)

    train2DataFilePath = train2DataPath + 'images_ids.txt'
    with open(train2DataFilePath, 'w') as train2IdsFile:
        create(images, labels, sourceFolderPath, train2DataPath, train2IdsFile, train2Indexes)

    valDataFilePath = valDataPath + 'images_ids.txt'
    with open(valDataFilePath, 'w') as valIdsFile:
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


def mergeFolders(folderList, outLoc):
    try:
        makedirs(outLoc)
    except:
        pass
    counter = 1
    for folder in folderList:
        allFiles = listdir(folder)
        images = []
        labels = []
        for file in allFiles:
            if file.startswith(PREFIX) and file.endswith(IMAGE_SUFFIX + ORIGIN_IMG_TYPE):
                images.append(file)
            elif file.startswith(PREFIX) and file.endswith(LABEL_SUFFIX + ORIGIN_LBL_TYPE):
                labels.append(file)

        images.sort()
        labels.sort()
        for img, lbl in zip(images, labels):
            image = np.array(im.open(folder + img))
            label = np.array(im.open(folder + lbl))
            imsave(outLoc + PREFIX + str(counter).zfill(6) + IMAGE_SUFFIX + ORIGIN_IMG_TYPE, image)
            imsave(outLoc + PREFIX + str(counter).zfill(6) + LABEL_SUFFIX + ORIGIN_LBL_TYPE, label)
            counter += 1


if __name__ == '__main__':
    # folderList = ['C:\\Git\\instance-seg\\cvppp\\training\\A1\\',
    #               'C:\\Git\\instance-seg\\cvppp\\training\\A2\\',
    #               'C:\\Git\\instance-seg\\cvppp\\training\\A3\\',
    #               'C:\\Git\\instance-seg\\cvppp\\training\\A4\\']

    # mergedLoc = 'C:\\Git\\instance-seg\\cvppp\\merged\\'
    # mergeFolders(folderList, mergedLoc)
    mergedLoc = 'C:\\Git\\instance-seg\\cvppp\\training\\A1\\'
    convert(mergedLoc, 'C:\\Git\\instance-seg\\cvppp\\formattedA1Only\\', 78,20)
