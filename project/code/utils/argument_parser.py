import os
import argparse
from datetime import datetime


def train_argument_parser():
    default_experiment_name = 'exp_' + str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    # default_train_data_path = os.path.join('..', '..', 'COCO', 'train2017', '')
    # default_train_data_path = os.path.join('..', '..', 'cvppp', 'all_plants', 'train1', 'images', '')
    default_train_data_path = os.path.join('..', '..', 'cvppp', 'formattedA1Only', 'train2', 'mask_distances_no_edges_no_weighted_mean_second_best', '')
    # default_train_data_path = os.path.join('..', '..', 'cvppp', 'formattedA1Only', 'train2',
    #                                        'mask_distances_original_paper', '')

    # default_train_labels_path = os.path.join('..', '..', 'COCO', 'train2017labels', 'instance_labels', '')
    # default_train_labels_path = os.path.join('..', '..', 'cvppp', 'all_plants', 'train1', 'labels', '')
    default_train_labels_path = os.path.join('..', '..', 'cvppp', 'formattedA1Only', 'train2', 'mask_labels_no_edges_no_weighted_mean_second_best', '')
    # default_train_labels_path = os.path.join('..', '..', 'cvppp', 'formattedA1Only', 'train2',
    #                                          'mask_labels_original_paper', '')

    # default_train_ids_file = os.path.join('..', '..', 'COCO', 'train2017labels', 'images_ids.txt')
    # default_train_ids_file = os.path.join('..', '..', 'cvppp', 'all_plants', 'train1', 'images_ids.txt')
    default_train_ids_file = os.path.join('..', '..', 'cvppp', 'formattedA1Only', 'train2', 'mask_ids_no_edges_no_weighted_mean_second_best.txt')
    # default_train_ids_file = os.path.join('..', '..', 'cvppp', 'all_plants', 'train2', 'mask_ids.txt')
    # default_train_ids_file = os.path.join('..', '..', 'cvppp', 'formattedA1Only', 'train2',
    #                                       'mask_ids_original_paper.txt')

    # default_train_ids_file = os.path.join('..', '..', 'COCO', 'overfit.txt')
    # default_train_ids_file = os.path.join('..', '..', 'cvppp', 'formatted', 'train2', 'overfit.txt')

    parser = argparse.ArgumentParser()
    parser.add_argument('--current_experiment', help='Experiment name', required=False, default=default_experiment_name)
    parser.add_argument('--train_data_folder_path', required=False, default=default_train_data_path)
    parser.add_argument('--train_labels_folder_path', required=False, default=default_train_labels_path)
    parser.add_argument('--train_ids_file_path', required=False, default=default_train_ids_file)
    parser.add_argument('--GPUs', required=False, type=str)

    args = parser.parse_args()
    current_experiment = args.current_experiment
    train_data_folder_path = args.train_data_folder_path
    train_labels_folder_path = args.train_labels_folder_path
    train_ids_path = args.train_ids_file_path

    GPUs = args.GPUs

    return current_experiment, train_data_folder_path, train_labels_folder_path, train_ids_path, GPUs


def validation_argument_parser():
    defaultExperimentName = 'exp_' + str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    # defaultDataPath = os.path.join('..', '..', 'COCO', 'train2017', '')
    # defaultLabelsPath = os.path.join('..', '..', 'COCO', 'train2017labels', 'instance_labels', '')
    # defaultIdsFile = os.path.join('..', '..', 'COCO', 'train2017labels', 'images_ids.txt')
    # defaultIdsFile = os.path.join('..', '..', 'COCO', 'overfit.txt')

    # defaultDataPath = os.path.join('..', '..', 'COCO', 'val2017', '')
    # defaultLabelsPath = os.path.join('..', '..', 'COCO', 'val2017labels', 'instance_labels', '')
    # defaultIdsFile = os.path.join('..', '..', 'COCO', 'val2017labels', 'images_ids.txt')

    # defaultDataPath = os.path.join('..', '..', 'cvppp', 'formatted', 'train', 'images', '')
    # defaultLabelsPath = os.path.join('..', '..', 'cvppp', 'formatted', 'train', 'labels', '')
    # defaultIdsFile = os.path.join('..', '..', 'cvppp', 'formatted', 'train', 'images_ids.txt')
    # defaultIdsFile = os.path.join('..', '..', 'cvppp', 'formatted', 'train', 'overfit.txt')

    defaultDataPath = os.path.join('..', '..', 'cvppp', 'formatted', 'train2', 'images', '')
    defaultLabelsPath = os.path.join('..', '..', 'cvppp', 'formatted', 'train2', 'labels', '')
    defaultIdsFile = os.path.join('..', '..', 'cvppp', 'formatted', 'train2', 'images_ids.txt')
    # defaultIdsFile = os.path.join('..', '..', 'cvppp', 'formatted', 'train2', 'overfit.txt')

    # defaultDataPath = os.path.join('..', '..', 'VOC2012', 'JPEGImages', '')
    # defaultLabelsPath = os.path.join('..', '..', 'VOC2012', 'SegmentationObject', '')
    # defaultIdsFile = os.path.join('..', '..', 'VOC2012', 'ImageSets', 'Segmentation', 'val.txt')

    parser = argparse.ArgumentParser()
    parser.add_argument('--current_experiment', help='Experiment name', required=False, default=defaultExperimentName)
    parser.add_argument('--epoch_num', help='Epoch number', required=False, default='latest')
    parser.add_argument('--data_folder_path', required=False, default=defaultDataPath)
    parser.add_argument('--labels_folder_path', required=False, default=defaultLabelsPath)
    parser.add_argument('--ids_file_path', required=False, default=defaultIdsFile)
    parser.add_argument('--GPUs', required=False, type=str)

    args = parser.parse_args()
    current_experiment = args.current_experiment
    dataPath = args.data_folder_path
    labelsPath = args.labels_folder_path
    idsPath = args.ids_file_path
    currentEpoch = args.epoch_num
    GPUs = args.GPUs

    # current_experiment = 'leafs_batch_5_cyc_lr_no_edges'
    # currentEpoch = '1001'

    return current_experiment, currentEpoch, dataPath, labelsPath, idsPath, GPUs


def evaluation_argument_parser():
    defaultFeExperimentName = 'exp_fe_' + str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    defaultClExperimentName = 'exp_cl_' + str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    defaultDataPath = os.path.join('..', '..', 'cvppp', 'formattedA1Only', 'val', 'images', '')
    defaultLabelsPath = os.path.join('..', '..', 'cvppp', 'formattedA1Only', 'val', 'labels', '')
    defaultIdsFile = os.path.join('..', '..', 'cvppp', 'formattedA1Only', 'val', 'images_ids.txt')
    defaultOutPath = os.path.join('.', 'outputs', '')

    parser = argparse.ArgumentParser()
    parser.add_argument('--fe_name', help='Feature extractor experiment name', required=False,
                        default=defaultFeExperimentName)
    parser.add_argument('--fe_sub_name', help='Feature extractor sub-experiment name', required=False, default='')
    parser.add_argument('--cl_name', help='Clustering experiment name', required=False, default=defaultClExperimentName)
    parser.add_argument('--cl_sub_name', help='Clustering sub-experiment name', required=False, default='')
    parser.add_argument('--fe_epoch_num', help='Feature extractor epoch number', required=False, default='latest')
    parser.add_argument('--cl_epoch_num', help='Clustering epoch number', required=False, default='latest')
    parser.add_argument('--data_folder_path', required=False, default=defaultDataPath)
    parser.add_argument('--labels_folder_path', required=False, default=defaultLabelsPath)
    parser.add_argument('--ids_file_path', required=False, default=defaultIdsFile)
    parser.add_argument('--output_path', required=False, default=defaultOutPath)
    parser.add_argument('--GPUs', required=False, type=str)

    args = parser.parse_args()
    fe_experiment = args.fe_name
    fe_sub_experiment = args.fe_sub_name
    cl_experiment = args.cl_name
    cl_sub_experiment = args.cl_sub_name
    feEpoch = args.fe_epoch_num
    clEpoch = args.cl_epoch_num
    dataPath = args.data_folder_path
    labelsPath = args.labels_folder_path
    idsPath = args.ids_file_path
    outputPath = args.output_path
    GPUs = args.GPUs

    return fe_experiment, fe_sub_experiment, cl_experiment, cl_sub_experiment, feEpoch, clEpoch, dataPath, labelsPath, idsPath, outputPath, GPUs
