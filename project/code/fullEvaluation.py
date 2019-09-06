from costum_dataset import CostumeDataset
from torch.utils.data import DataLoader
from config import getFeatureExtractionModel, getClusterModel, config_logger, float_type, long_type
from matplotlib.colors import ListedColormap
from matplotlib.pyplot import imsave
from os.path import join
from os import makedirs
import matplotlib
import torch
import numpy as np
from prediction import cluster_features
from mrf_wrapper import denoise_colored_image
from evaluate import Evaluator
import skimage
from skimage.transform import rescale

matplotlib.use('Agg')
FILE_NAME_ID_LENGTH = 4
IMAGE_FORMAT = 'jpg'
LABELS_FORMAT = 'png'

MIN_CLUSTER_SIZE = 10
RESCALE_FACTOR = 2

useMrfAfterHdbScan = False  # True
useClusteringNet = False  # True
useMrfAfterClusteringNet = False  # True


def downsample(image, factor):
    image = skimage.measure.block_reduce(image, (factor, factor), np.max)  # reduce resolution for performance
    return image


def upsample(image, factor):
    image = rescale(image, order=0, mode='constant', scale=factor,
                    preserve_range=True)
    return image


def getMeanTensor(features, labels, focusLabel):
    labels = labels.flatten()
    features = features.permute(1, 2, 0).contiguous()
    shape = features.size()
    features = features.view(shape[0] * shape[1], shape[2])
    locations = torch.LongTensor(np.where(labels == focusLabel)[0]).type(long_type)
    # all vectors of this instance
    vectors = torch.index_select(features, dim=0, index=locations).type(float_type)
    meanTensor = vectors.mean(dim=0)
    return meanTensor


def getDistancesMask(features, centerTensor):
    distMap = (features.permute(1, 2, 0).type(float_type) - centerTensor.type(float_type)).norm(dim=2, p=2)
    distMap = distMap / distMap.max()
    distMap = 1 - distMap
    return distMap


def getColorMap():
    myCmap = [
        [0, 0, 0],
        [0.988235294117647, 0.913725490196078, 0.309803921568627],
        [0.447058823529412, 0.623529411764706, 0.811764705882353],
        [0.937254901960784, 0.160784313725490, 0.160784313725490],
        [0.678431372549020, 0.498039215686275, 0.658823529411765],
        [0.541176470588235, 0.886274509803922, 0.203921568627451],
        [0.913725490196078, 0.725490196078431, 0.431372549019608],
        [0.988235294117647, 0.686274509803922, 0.243137254901961],
        [0.827450980392157, 0.843137254901961, 0.811764705882353],
        [0.768627450980392, 0.627450980392157, 0],
        [0.125490196078431, 0.290196078431373, 0.529411764705882],
        [0.643137254901961, 0, 0],
        [0.360784313725490, 0.207843137254902, 0.400000000000000],
        [0.305882352941177, 0.603921568627451, 0.0235294117647059],
        [0.560784313725490, 0.349019607843137, 0.00784313725490196],
        [0.807843137254902, 0.360784313725490, 0],
        [0.533333333333333, 0.541176470588235, 0.521568627450980],
        [0.929411764705882, 0.831372549019608, 0],
        [0.203921568627451, 0.396078431372549, 0.643137254901961],
        [0.800000000000000, 0, 0],
        [0.458823529411765, 0.313725490196078, 0.482352941176471],
        [0.450980392156863, 0.823529411764706, 0.0862745098039216],
        [0.756862745098039, 0.490196078431373, 0.0666666666666667],
        [0.960784313725490, 0.474509803921569, 0],
        [0.729411764705882, 0.741176470588235, 0.713725490196078],
        [0.333333333333333, 0.341176470588235, 0.325490196078431],
        [0.180392156862745, 0.203921568627451, 0.211764705882353],
        [0.933333333333333, 0.933333333333333, 0.925490196078431],
        [0, 0, 0.0392156862745098],
        [0.988235294117647, 0.913725490196078, 0.349019607843137],
        [0.447058823529412, 0.623529411764706, 0.850980392156863],
        [0.937254901960784, 0.160784313725490, 0.200000000000000],
        [0.678431372549020, 0.498039215686275, 0.698039215686275],
        [0.541176470588235, 0.886274509803922, 0.243137254901961],
        [0.913725490196078, 0.725490196078431, 0.470588235294118],
        [0.988235294117647, 0.686274509803922, 0.282352941176471],
        [0.827450980392157, 0.843137254901961, 0.850980392156863],
        [0.768627450980392, 0.627450980392157, 0.0392156862745098],
        [0.125490196078431, 0.290196078431373, 0.568627450980392],
        [0.643137254901961, 0, 0.0392156862745098],
        [0.360784313725490, 0.207843137254902, 0.439215686274510],
        [0.305882352941177, 0.603921568627451, 0.0627450980392157],
        [0.560784313725490, 0.349019607843137, 0.0470588235294118],
        [0.807843137254902, 0.360784313725490, 0.0392156862745098],
        [0.533333333333333, 0.541176470588235, 0.560784313725490],
        [0.929411764705882, 0.831372549019608, 0.0392156862745098],
        [0.203921568627451, 0.396078431372549, 0.682352941176471],
        [0.800000000000000, 0, 0.0392156862745098],
        [0.458823529411765, 0.313725490196078, 0.521568627450980],
        [0.450980392156863, 0.823529411764706, 0.125490196078431],
        [0.756862745098039, 0.490196078431373, 0.105882352941176],
        [0.960784313725490, 0.474509803921569, 0.0392156862745098],
        [0.729411764705882, 0.741176470588235, 0.752941176470588],
        [0.333333333333333, 0.341176470588235, 0.364705882352941],
        [0.180392156862745, 0.203921568627451, 0.250980392156863],
        [0.933333333333333, 0.933333333333333, 0.964705882352941],
    ]
    return ListedColormap(myCmap)


def saveImage(outputDir, name, id, data):
    if len(data.shape) == 4:
        data = data[0]
    imsave(join(outputDir, str(id).zfill(FILE_NAME_ID_LENGTH) + '_' + name + '.' + IMAGE_FORMAT), data,
           format=IMAGE_FORMAT)


def saveLabel(outputDir, name, id, data):
    if len(data.shape) == 3:
        data = data[0]
    imsave(join(outputDir, str(id).zfill(FILE_NAME_ID_LENGTH) + '_' + name + '.' + LABELS_FORMAT), data,
           cmap=getColorMap(),
           format=LABELS_FORMAT)


def getClusters(features):
    '''
    cluster the features using HDBScan
    :param features: ndarray of shape (c,h,w) or (1,c,h,w)
    :return: ndarray of shape (h,w). each element is labeled
    '''
    if len(features.shape) == 4:
        features = features[0]
    features = np.transpose(features, (1, 2, 0))
    h = features.shape[0]
    w = features.shape[1]
    c = features.shape[2]
    features = np.reshape(features, [h * w, c])
    predicted = cluster_features(features, MIN_CLUSTER_SIZE)
    predicted = np.reshape(predicted, [h, w])
    return predicted


def convertToClusterNetInput(features, labels):
    '''
    convert the features outputted from the feature extractor to inputs for the ClusterNet
    :param features: torch.Tensor of shape (1,c,h,w) or (c,h,w)
    :param labeles: label estimate for each pixel, ndarray of shape (h,w) or (1,h,w)
    :return: torch.Tensor of shape (N,1,h,w), the slice [i,0,:,:] is the i'th labels distances
     (N is the number of instances without background)
    '''
    if len(features.shape) == 4:
        features = features[0]
    if len(labels.shape) == 3:
        labels = labels[0]
    h = labels.shape[0]
    w = labels.shape[1]
    colors, counts = np.unique(labels, return_counts=True)
    countSortedInd = np.argsort(-counts)  # get sorted indices from big to small
    colors = colors[countSortedInd]
    counts = counts[countSortedInd]
    backgroundColor = colors[np.argmax(counts)]
    assert (backgroundColor == colors[0])

    N = len(colors) - 1  # without background
    converted = torch.zeros((N, 1, h, w)).type(float_type)
    loc = 0
    for color in colors:
        if color == backgroundColor:
            continue
        meanTensor = getMeanTensor(features, labels, color)
        distanceMask = getDistancesMask(features, meanTensor)
        converted[loc, 0] = distanceMask
        loc += 1
    return converted


def convertIndividualSegmentsToSingleImage(segments):
    '''
    converts the outputs of the ClusterNet to single labeled image
    :param segments: ndarray of shape (N,1,h,w) each element is 0 or 1 for N non-background instances
    :return: ndarray of shape (h,w) each pixel has the a value of either {0,1,...,N} for some instance or background
    '''
    N = segments.shape[0]
    h = segments.shape[2]
    w = segments.shape[3]

    converted = np.zeros((h, w))
    currLabel = 1

    for i in range(N):
        currSegment = segments[i, 0]
        updateLocations = np.where(currSegment > 0.5)
        converted[updateLocations] = currLabel  # in case of a collision the last segment wins
        currLabel += 1

        # uncomment the following instead of the previous three lines in order to create two segments in the collision

        # newUpdateLocations = np.where((currSegment > 0.5) & (converted == 0))  # all unlabeled locations in this segment
        # labelToAvoid = 0
        # if newUpdateLocations[0].size > 0:
        #     converted[newUpdateLocations] = currLabel
        #     labelToAvoid = currLabel
        #     currLabel += 1
        # reUpdateLocations = np.where(
        #     (currSegment > 0.5) & (converted != labelToAvoid))  # all previously labeled locations in this segement
        #
        # if reUpdateLocations[0].size > 0:
        #     converted[reUpdateLocations] = currLabel
        #     currLabel += 1

    return converted


def run(feExpName, feSubName, clExpName, clSubName, feEpoch, clEpoch, dataPath, labelsPath, idsFilePath, outputPath):
    makedirs(outputPath, exist_ok=True)
    dataset = CostumeDataset(idsFilePath, dataPath, labelsPath)
    dataLoader = DataLoader(dataset, batch_size=1, shuffle=False)
    if useClusteringNet:
        loggerExpName = 'evaluation_' + feExpName + '_' + feSubName + '_' + clExpName + '_' + clSubName
    else:
        loggerExpName = 'evaluation_' + feExpName + '_' + feSubName
    logger = config_logger(loggerExpName)
    featureExtractorModel = \
        getFeatureExtractionModel(feExpName, logger, sub_experiment_name=feSubName, currentEpoch=feEpoch)[0]
    featureExtractorModel.eval()
    if useClusteringNet:
        clusteringModel = getClusterModel(clExpName, logger, sub_experiment_name=clSubName, currentEpoch=clEpoch)[0]
        clusteringModel.eval()
    hdbEval = Evaluator()
    hdbEvalResults = []
    hdbMrfEval = Evaluator()
    hdbMrfEvalResults = []
    hdbClusterNetEval = Evaluator()
    hdbClusterNetEvalResults = []
    hdbClusterNetMrfEval = Evaluator()
    hdbClusterNetMrfEvalResults = []
    hdbMrfClusterNetEval = Evaluator()
    hdbMrfClusterNetEvalResults = []
    hdbMrfClusterNetMrfEval = Evaluator()
    hdbMrfClusterNetMrfEvalResults = []
    for i, batch in enumerate(dataLoader):
        inputs = batch['image'].type(float_type)
        labels = batch['label'].cpu().numpy()
        labels = labels[0]

        saveImage(outputPath, 'image', i, batch['originalImage'].cpu().numpy())
        saveLabel(outputPath, 'ground_truth', i, labels)

        features = featureExtractorModel(inputs, None, None)[0]

        clustered = getClusters(features.cpu().numpy())
        saveLabel(outputPath, 'hdbscan', i, clustered)
        hdbEvalResults.append(hdbEval.evaluate(clustered, labels))

        if useMrfAfterHdbScan:
            clusteredAndMRF = upsample(denoise_colored_image(downsample(clustered, RESCALE_FACTOR)), RESCALE_FACTOR)
            saveLabel(outputPath, 'hdbscan_mrf', i, clusteredAndMRF)
            hdbMrfEvalResults.append(hdbMrfEval.evaluate(clusteredAndMRF, labels))

        if useClusteringNet:
            clusteredInput = convertToClusterNetInput(features, clustered)
            if clusteredInput.shape[0] > 0:
                noMrfOnInput = clusteringModel(clusteredInput, None)[0]
            else:
                noMrfOnInput = np.zeros((1, 1, clusteredInput.shape[2], clusteredInput.shape[3]))
            hdbClusterNetOut = convertIndividualSegmentsToSingleImage(noMrfOnInput)
            saveLabel(outputPath, 'hdbscan_clusternet', i, hdbClusterNetOut)
            hdbClusterNetEvalResults.append(hdbClusterNetEval.evaluate(hdbClusterNetOut, labels))

        if useMrfAfterHdbScan and useClusteringNet:
            clusteredMrfInput = convertToClusterNetInput(features, clusteredAndMRF)
            if clusteredMrfInput.shape[0] > 0:
                mrfOnInput = clusteringModel(clusteredMrfInput, None)[0]
            else:
                mrfOnInput = np.zeros((1, 1, clusteredMrfInput.shape[2], clusteredMrfInput.shape[3]))
            hdbMrfClusterNetOut = convertIndividualSegmentsToSingleImage(mrfOnInput)
            saveLabel(outputPath, 'hdbscan_mrf_clusternet', i, hdbMrfClusterNetOut)
            hdbMrfClusterNetEvalResults.append(hdbMrfClusterNetEval.evaluate(hdbMrfClusterNetOut, labels))

        if useClusteringNet and useMrfAfterClusteringNet:
            hdbClusterNetMrfOut = upsample(denoise_colored_image(downsample(hdbClusterNetOut, RESCALE_FACTOR)),
                                           RESCALE_FACTOR)
            saveLabel(outputPath, 'hdbscan_clusternet_mrf', i, hdbClusterNetMrfOut)
            hdbClusterNetMrfEvalResults.append(hdbClusterNetMrfEval.evaluate(hdbClusterNetMrfOut, labels))

        if useMrfAfterHdbScan and useClusteringNet and useMrfAfterClusteringNet:
            hdbMrfClusterNetMrfOut = upsample(denoise_colored_image(downsample(hdbMrfClusterNetOut, RESCALE_FACTOR)),
                                              RESCALE_FACTOR)
            saveLabel(outputPath, 'hdbscan_mrf_clusternet_mrf', i, hdbMrfClusterNetMrfOut)
            hdbMrfClusterNetMrfEvalResults.append(hdbMrfClusterNetMrfEval.evaluate(hdbMrfClusterNetMrfOut, labels))

    hdbEvalResults.append(hdbEval.get_average_results())

    if useMrfAfterHdbScan:
        hdbMrfEvalResults.append(hdbMrfEval.get_average_results())

    if useClusteringNet:
        hdbClusterNetEvalResults.append(hdbClusterNetEval.get_average_results())

    if useMrfAfterHdbScan and useClusteringNet:
        hdbMrfClusterNetEvalResults.append(hdbMrfClusterNetEval.get_average_results())

    if useClusteringNet and useMrfAfterClusteringNet:
        hdbClusterNetMrfEvalResults.append(hdbClusterNetMrfEval.get_average_results())

    if useMrfAfterHdbScan and useClusteringNet and useMrfAfterClusteringNet:
        hdbMrfClusterNetMrfEvalResults.append(hdbMrfClusterNetMrfEval.get_average_results())

    with open(join(outputPath, 'statistics.txt'), mode='w') as file:
        for i in range(len(hdbEvalResults) - 1):
            file.write('hdbscan only image ' + str(i).zfill(FILE_NAME_ID_LENGTH) + ': ' + str(hdbEvalResults[i]))
            file.write('\n')
            if useMrfAfterHdbScan:
                file.write(
                    'hdbscan and MRF image ' + str(i).zfill(FILE_NAME_ID_LENGTH) + ': ' + str(hdbMrfEvalResults[i]))
                file.write('\n')
            if useClusteringNet:
                file.write('hdbscan and ClusterNet image ' + str(i).zfill(FILE_NAME_ID_LENGTH) + ': ' + str(
                    hdbClusterNetEvalResults[i]))
                file.write('\n')
            if useMrfAfterHdbScan and useClusteringNet:
                file.write('hdbscan and MRF and ClusterNet image ' + str(i).zfill(FILE_NAME_ID_LENGTH) + ': ' + str(
                    hdbMrfClusterNetEvalResults[i]))
                file.write('\n')
            if useClusteringNet and useMrfAfterClusteringNet:
                file.write('hdbscan and ClusterNet and MRF image ' + str(i).zfill(FILE_NAME_ID_LENGTH) + ': ' + str(
                    hdbClusterNetMrfEvalResults[i]))
                file.write('\n')
            if useMrfAfterHdbScan and useClusteringNet and useMrfAfterClusteringNet:
                file.write(
                    'hdbscan and MRF and ClusterNet and MRF image ' + str(i).zfill(FILE_NAME_ID_LENGTH) + ': ' + str(
                        hdbMrfClusterNetMrfEvalResults[i]))
                file.write('\n')
            file.write('\n')
        lastLoc = len(hdbEvalResults) - 1

        file.write('hdbscan only mean: ' + str(hdbEvalResults[lastLoc]))
        file.write('\n')
        if useMrfAfterHdbScan:
            file.write('hdbscan and MRF mean: ' + str(hdbMrfEvalResults[lastLoc]))
            file.write('\n')
        if useClusteringNet:
            file.write('hdbscan and ClusterNet mean : ' + str(hdbClusterNetEvalResults[lastLoc]))
            file.write('\n')
        if useMrfAfterHdbScan and useClusteringNet:
            file.write('hdbscan and MRF and ClusterNet mean: ' + str(hdbMrfClusterNetEvalResults[lastLoc]))
            file.write('\n')
        if useClusteringNet and useMrfAfterClusteringNet:
            file.write('hdbscan and ClusterNet and MRF mean: ' + str(hdbClusterNetMrfEvalResults[lastLoc]))
            file.write('\n')
        if useMrfAfterHdbScan and useClusteringNet and useMrfAfterClusteringNet:
            file.write('hdbscan and MRF and ClusterNet and MRF mean: ' + str(hdbMrfClusterNetMrfEvalResults[lastLoc]))
            file.write('\n')
