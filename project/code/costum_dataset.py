import os

from torch.utils.data import Dataset
import PIL.Image as im
import numpy as np
from torchvision import transforms
from config import PIXEL_IGNORE_VAL
from config import PIXEL_BOUNDARY_VAL

from augmentation import augmentation_func


class CostumeDataset(Dataset):
    '''
    use this for training only, as the images are cropped to fit the network size.
    :param ids_file_path - path to a file containing the ids of all the images, i.e. the
               file name of each image - for example "1234.jpg" will be represented as "1234\n".
    :param data_path - path to the directory containing the jpeg images.
    :param labels_path - a path to the directory containing the labels. Labels are PASCAL VOC style
                        .png images, containing instance segmentations.
    :param img_h, img_w - images are rescaled and cropped to this size.
    '''

    def __init__(self, ids_file_path, data_path, labels_path, mode="val", img_h=224, img_w=224):
        ids_file = open(ids_file_path)
        self.ids = ids_file.read().split("\n")[:-1]

        self.data_path = data_path
        self.labels_path = labels_path
        self.h = img_h
        self.w = img_w
        self.mode = mode

        self.data_transforms = {
            "default": transforms.Compose([
                ResizeSample(self.h, self.w)]
            ),
            "augmentation": augmentation_func,
            "img": transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            "label": transforms.Compose([
                AsArray()
            ])
        }

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        id = self.ids[item]
        img = im.open(os.path.join(self.data_path, id + '.jpg')).convert('RGB')
        label = im.open(os.path.join(self.labels_path, id + '.png'))

        size = label.size

        img = self.data_transforms["default"]((img, False))
        label = self.data_transforms["default"]((label, True))

        if self.mode == "train":
            augmentation = self.data_transforms["augmentation"]
            # apply the same augmentation on image and label
            img, label = augmentation(img, label)

        originalImage = self.data_transforms['label'](img)
        img = self.data_transforms["img"](img)
        label = self.data_transforms["label"](label)

        labelEdges = label.copy()
        for w in range(self.w):
            for h in range(self.h):
                if not self.__isBoundaryPixel(w, h, label):
                    labelEdges[w, h] = PIXEL_IGNORE_VAL  # this is special value to ignore

        return {'image': img, 'originalImage': originalImage, 'label': label, 'labelEdges': labelEdges, 'size': size}

    def __isBoundaryPixel(self, w, h, labelIm):
        pixelVal = labelIm[w, h]
        if pixelVal == PIXEL_IGNORE_VAL or pixelVal == PIXEL_BOUNDARY_VAL:
            return False
        labelShape = labelIm.shape
        if h > 0:  # not the top row
            if pixelVal != labelIm[w, h - 1]:
                return True
        if h < labelShape[1] - 1:  # not the bottom row
            if pixelVal != labelIm[w, h + 1]:
                return True
        if w > 0:  # not the most left column
            if pixelVal != labelIm[w - 1, h]:
                return True
        if w < labelShape[0] - 1:  # not the most right column
            if pixelVal != labelIm[w + 1, h]:
                return True
        if h > 0 and w > 0:  # not in the top row and not in the left most column
            if pixelVal != labelIm[w - 1, h - 1]:
                return True
        if h > 0 and w < labelShape[0] - 1:  # not in the top row and not in the right most column
            if pixelVal != labelIm[w + 1, h - 1]:
                return True
        if h < labelShape[1] - 1 and w > 0:  # not in the bottom row and not in the left most column
            if pixelVal != labelIm[w - 1, h + 1]:
                return True
        if h < labelShape[1] - 1 and w < labelShape[0] - 1:  # not in the bottom row and not in the right most column
            if pixelVal != labelIm[w + 1, h + 1]:
                return True
        return False


class ResizeSample(object):
    '''
    utility transformation to resize sample(PIL image and label) to a given dimension
    without cropping information. the network takes in tensors with dimensions
    that are multiples of 32.
    :param img: PIL image to resize
    :param label: PIL image with the label to resize
    :param h: desired height
    :param w: desired width
    :param restore: set this to true when you want to restore a padded image to it's
                    original dimensions
    :param evaluate: if set to True, images are rescaled on the long side, and padded.
                        if False, images are rescaled on the short side and cropped.
    :return: the resized image, label
    '''

    def __init__(self, h, w, restore=False, evaluate=False):
        self.h = h
        self.w = w
        self.restore = restore
        self.evaluate = evaluate

    def __call__(self, imgTuple):
        img = imgTuple[0]
        isLabel = imgTuple[1]
        center_crop = transforms.CenterCrop([self.h, self.w])

        old_size = img.size  # old_size is in (width, height) format
        w_ratio = float(self.w) / old_size[0]
        h_ratio = float(self.h) / old_size[1]
        if self.restore or not self.evaluate:
            ratio = max(w_ratio, h_ratio)
        else:
            ratio = min(w_ratio, h_ratio)
        new_size = tuple([int(x * ratio) for x in old_size])

        if not isLabel:
            img = img.resize(new_size, im.ANTIALIAS)
        else:
            img = img.resize(new_size, im.NEAREST)  # for labels use NEAREST to avoid adding "new" labels

        img = center_crop(img)

        return img


class AsArray(object):
    def __call__(self, img):
        return np.asarray(img)
