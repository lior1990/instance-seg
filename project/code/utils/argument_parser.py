import os
import argparse
from datetime import datetime


def train_argument_parser():
    default_experiment_name = 'exp_' + str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    # default_train_data_path = os.path.join('..', '..', 'COCO', 'train2017', '')
    default_train_data_path = os.path.join('..', '..', 'cvppp', 'formatted', 'train', 'images', '')

    # default_train_labels_path = os.path.join('..', '..', 'COCO', 'train2017labels', 'instance_labels', '')
    default_train_labels_path = os.path.join('..', '..', 'cvppp', 'formatted', 'train', 'labels', '')

    # default_train_ids_file = os.path.join('..', '..', 'COCO', 'train2017labels', 'images_ids.txt')
    default_train_ids_file = os.path.join('..', '..', 'cvppp', 'formatted', 'train', 'images_ids.txt')

    # default_train_ids_file = os.path.join('..', '..', 'COCO', 'overfit.txt')
    default_train_ids_file = os.path.join('..', '..', 'cvppp', 'formatted', 'train', 'overfit.txt')

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

    defaultDataPath = os.path.join('..', '..', 'cvppp', 'formatted', 'val', 'images', '')
    defaultLabelsPath = os.path.join('..', '..', 'cvppp', 'formatted', 'val', 'labels', '')
    defaultIdsFile = os.path.join('..', '..', 'cvppp', 'formatted', 'val', 'images_ids.txt')

    parser = argparse.ArgumentParser()
    parser.add_argument('--current_experiment', help='Experiment name', required=False, default=defaultExperimentName)
    parser.add_argument('--epoch_num',help='Epoch number',required=False,default='latest')
    parser.add_argument('--data_folder_path', required=False, default=defaultDataPath)
    parser.add_argument('--labels_folder_path', required=False, default=defaultLabelsPath)
    parser.add_argument('--ids_file_path', required=False, default=defaultIdsFile)

    args = parser.parse_args()
    current_experiment = args.current_experiment
    dataPath = args.data_folder_path
    labelsPath = args.labels_folder_path
    idsPath = args.ids_file_path
    currentEpoch = args.epoch_num

    return current_experiment, currentEpoch, dataPath, labelsPath, idsPath
