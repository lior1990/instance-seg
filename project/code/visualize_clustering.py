import os
import numpy as np

from sklearn.manifold import TSNE
from utils.argument_parser import validation_argument_parser
import torch.autograd
from costum_dataset import CostumeDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from config import config_experiment, float_type
from matplotlib import pyplot as plt


from utils.model_loader import load_model_from_experiment


def run(current_experiment,currentEpoch, data_path, labels_path, ids_path):
    try:
        os.makedirs(os.path.join('cluster_visualizations', current_experiment))
    except:
        pass

    dataset = CostumeDataset(ids_path, data_path, labels_path, img_h=224, img_w=224)
    dataloader = DataLoader(dataset)

    # Set up an experiment
    experiment, exp_logger = config_experiment(current_experiment, resume=True, useBest=False,currentEpoch=currentEpoch)

    fe = load_model_from_experiment(experiment)

    if torch.cuda.is_available():
        print("Using CUDA")
        fe.cuda()

    fe.eval()
    for i,batch in enumerate(dataloader):
        try:
            inputs = Variable(batch['image'].type(float_type))
            features, _ = fe(inputs, None, None)
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

    current_experiment, currentEpoch, dataPath, labelsPath, idsPath = validation_argument_parser()

    with torch.no_grad():
        run(current_experiment, currentEpoch, dataPath, labelsPath, idsPath)


if __name__ == '__main__':
    main()
