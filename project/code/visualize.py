import argparse
from collections import OrderedDict

import torch.autograd
from costum_dataset import *
from torch.utils.data import DataLoader
from evaluate import *
from config import *
from datetime import datetime
from ModelWithLoss import CompleteModel

import os


def run(current_experiment,currentEpoch, data_path, labels_path, ids_path):

    dataset = CostumeDataset(ids_path, data_path, labels_path, img_h=224, img_w=224)
    dataloader = DataLoader(dataset)

    # Set up an experiment
    experiment, exp_logger = config_experiment(current_experiment, resume=True, useBest=False,currentEpoch=currentEpoch)

    fe = CompleteModel(embedding_dim)

    try:
        fe.load_state_dict(experiment['fe_state_dict'])
    except:
        state_dict = OrderedDict()
        prefix = 'module.'
        for key,val in experiment['fe_state_dict'].items():
            if key.startswith(prefix):
                key = key[len(prefix):]
            state_dict[key] = val
        fe.load_state_dict(state_dict)




    if torch.cuda.is_available():
        print("Using CUDA")
        fe.cuda()

    fe.eval()
    for i,batch in enumerate(dataloader):
        try:
            inputs = Variable(batch['image'].type(float_type))
            labels = batch['label'].cpu().numpy()
            features,xxx = fe(inputs,None,None)
            inputs = inputs.cpu().numpy().squeeze()
            features  = features.cpu().numpy().squeeze()
            labels = labels.squeeze()
            visualize(inputs,labels,features,current_experiment,i)
        except:
            continue
    return




def main():

    defaultExperimentName = 'exp_' + str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    defaultDataPath = os.path.join('..', '..', 'COCO', 'train2017', '')
    defaultLabelsPath = os.path.join('..', '..', 'COCO', 'train2017labels', 'instance_labels', '')
    defaultIdsFile = os.path.join('..', '..', 'COCO', 'train2017labels', 'images_ids.txt')
    defaultIdsFile = os.path.join('..', '..', 'COCO', 'overfit.txt')
    #
    # defaultDataPath = os.path.join('..', '..', 'COCO', 'val2017', '')
    # defaultLabelsPath = os.path.join('..', '..', 'COCO', 'val2017labels', 'instance_labels', '')
    # defaultIdsFile = os.path.join('..', '..', 'COCO', 'val2017labels', 'images_ids.txt')



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



    current_experiment = 'exp_overfit'
    currentEpoch = str(1)
    with torch.no_grad():
        run(current_experiment,currentEpoch, dataPath, labelsPath, idsPath)


if __name__ == '__main__':
    main()
