import os
import numpy as np
import matplotlib

matplotlib.use('Agg')

import torch.autograd
from imageio import imsave
# from scipy.misc import imsave
from matplotlib import pyplot as plt
from costum_dataset import CostumeDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from config import config_logger,getFeatureExtractionModel, float_type
from prediction import predict_label


def visualize(input, label, features, name, id):
    '''
    This function performs postprocessing (dimensionality reduction and clustering) for a given network
    output. it also visualizes the resulted segmentation along with the original image and the ground truth
    segmentation and saves all the images locally.
    :param input: (3, h, w) ndarray containing rgb data as outputted by the costume datasets
    :param label: (h, w) or (1, h, w) ndarray with the ground truth segmentation
    :param features: (c, h, w) ndarray with the embedded pixels outputted by the network
    :param name: str with the current experiment name
    :param id: an identifier for the current image (for file saving purposes)
    :return: None. all the visualizations are saved locally
    '''
    # Save original image
    os.path.join("visualizations", name, "segmentations")
    os.makedirs(os.path.join("visualizations", name, "segmentations"), exist_ok=True)
    img_data = np.transpose(input, [1, 2, 0])
    max_val = np.amax(np.absolute(img_data))
    img_data = (img_data/max_val + 1) / 2  # normalize img
    image_path = os.path.join("visualizations", name, "segmentations", str(id)+"img.jpg")
    imsave(image_path, img_data)

    # Save ground truth
    if len(label.shape)==3:
        label = np.squeeze(label)
    label[np.where(label==255)] = 0
    label = label.astype(np.int32)
    gt_path = os.path.join("visualizations", name, "segmentations", str(id)+"gt.jpg")
    imsave(gt_path, label)

    # reduce features dimensionality and predict label
    predicted_label = predict_label(features, downsample_factor=2)
    seg_path = os.path.join("visualizations", name, "segmentations", str(id)+"seg.jpg")
    imsave(seg_path, predicted_label)

    # draw predicted seg on img and save
    plt.imshow(img_data)
    plt.imshow(predicted_label, alpha=0.5)
    vis_path = os.path.join("visualizations", name, "segmentations", str(id)+"vis.jpg")
    plt.savefig(vis_path)
    plt.close()

    return


def run(current_experiment,currentEpoch, data_path, labels_path, ids_path):

    dataset = CostumeDataset(ids_path, data_path, labels_path, img_h=224, img_w=224)
    dataloader = DataLoader(dataset)

    # Set up an experiment
    logger = config_logger(current_experiment)
    fe = getFeatureExtractionModel(current_experiment,logger,currentEpoch=currentEpoch)[0]

    fe.eval()
    for i,batch in enumerate(dataloader):
        inputs = Variable(batch['image'].type(float_type))
        labels = batch['label'].cpu().numpy()
        results = fe(inputs,None,None)
        features = results[0]
        inputs = inputs.cpu().numpy().squeeze()
        features  = features.cpu().numpy().squeeze()
        labels = labels.squeeze()
        visualize(inputs,labels,features,current_experiment,i)

    return
