import os
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from config import chkpts_dir, embedding_dim, context, float_type
from costum_dataset import CostumeDataset
from evaluate import visualize
from model import FeatureExtractor

# load the model
exp_path = os.path.join(chkpts_dir, "first_test", "best.pth")
experiment = torch.load(exp_path, map_location=lambda storage, loc: storage)
feature_extractor = FeatureExtractor(embedding_dim, context=context)
feature_extractor.load_state_dict(experiment['fe_state_dict'])

# create test set
test_images_path = "C:\\Users\\Lior\\Downloads\\VOCdevkit\\VOC2012\\JPEGImages\\"
test_labels_path = "C:\\Users\\Lior\\Downloads\\VOCdevkit\\VOC2012\\SegmentationObject\\"
test_ids_path = "C:\\Users\\Lior\\Downloads\\VOCdevkit\\VOC2012\\ImageSets\\Segmentation\\train.txt"

#
# test_ids_path = "C:\\Users\\Lior\\git\\instance-seg\\test_images\\test.txt"
# test_images_path = "C:\\Users\\Lior\\git\\instance-seg\\test_images\\data\\"
# test_labels_path = "C:\\Users\\Lior\\git\\instance-seg\\test_images\\labels\\"
test_dataset = CostumeDataset(test_ids_path, test_images_path, test_labels_path, img_h=224, img_w=224)
test_dataloader = DataLoader(test_dataset)

for i, batch in enumerate(test_dataloader):
    inputs = Variable(batch['image'].type(float_type))
    labels = batch['label'].cpu().numpy()
    batch_features = feature_extractor(inputs)
    for input, label, features in zip(inputs, labels, batch_features):
        visualize(input.numpy(), label, features.detach().numpy(), "first_test", i)


if __name__ == '__main__':
    pass
