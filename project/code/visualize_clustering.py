from sklearn.manifold import TSNE
import argparse
import torch.autograd
from costum_dataset import *
from torch.utils.data import DataLoader
from evaluate import *
from config import *
from datetime import datetime
import MetricLearningModel

import os


def run(current_experiment, data_path, labels_path, ids_path):
    try:
        os.makedirs(os.path.join('cluster_visualizations', current_experiment))
    except:
        pass

    dataset = CostumeDataset(ids_path, data_path, labels_path, img_h=224, img_w=224)
    dataloader = DataLoader(dataset)

    # Set up an experiment
    experiment, exp_logger = config_experiment(current_experiment, resume=True, useBest=False)

    fe = MetricLearningModel.FeatureExtractor(embedding_dim)

    fe.load_state_dict(experiment['fe_state_dict'])

    if torch.cuda.is_available():
        print("Using CUDA")
        fe.cuda()

    fe.eval()
    for i,batch in enumerate(dataloader):
        try:
            inputs = Variable(batch['image'].type(float_type))
            features = fe(inputs)
            features = features.cpu().numpy().squeeze()
            features = np.transpose(features, [1, 2, 0])  # transpose to (h,w,c)

            labels = batch['label'].cpu().numpy()
            labels = labels.squeeze()
            flat_labels = labels.flatten()

            h = features.shape[0]
            w = features.shape[1]
            c = features.shape[2]

            flat_features = np.reshape(features, [h * w, c])

            # find tsne coords for 2 dimensions
            tsne = TSNE(n_components=2, random_state=0)
            np.set_printoptions(suppress=True)
            features_2d = tsne.fit_transform(flat_features)

            instances = np.unique(flat_labels)

            plt.figure(figsize=(6, 5))
            colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
            for idx, instance in enumerate(instances):
                plt.scatter(features_2d[flat_labels == instance, 0], features_2d[flat_labels == instance, 1], c=colors[idx], label=instance)
            plt.legend()
            plt.savefig(os.path.join('cluster_visualizations', current_experiment, "%s.png" % i))
            plt.close()
            print("Done %s" % i)
        except:
            continue
    return




def main():

    defaultExperimentName = 'exp_' + str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    defaultDataPath = os.path.join('..', '..', 'COCO', 'train2017', '')
    defaultLabelsPath = os.path.join('..', '..', 'COCO', 'train2017labels', 'instance_labels', '')
    defaultIdsFile = os.path.join('..', '..', 'COCO', 'train2017labels', 'images_ids.txt')

    parser = argparse.ArgumentParser()
    parser.add_argument('--current_experiment', help='Experiment name', required=False, default=defaultExperimentName)
    parser.add_argument('--data_folder_path', required=False, default=defaultDataPath)
    parser.add_argument('--labels_folder_path', required=False, default=defaultLabelsPath)
    parser.add_argument('--ids_file_path', required=False, default=defaultIdsFile)

    args = parser.parse_args()
    current_experiment = args.current_experiment
    dataPath = args.data_folder_path
    labelsPath = args.labels_folder_path
    idsPath = args.ids_file_path


    with torch.no_grad():
        run(current_experiment, dataPath, labelsPath, idsPath)


if __name__ == '__main__':
    main()
