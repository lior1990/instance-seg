import random

from torchvision.transforms import ColorJitter, Grayscale, RandomChoice
from torchvision.transforms.functional import hflip, vflip, rotate


def transform(image, label):
    image, label = transform_both(image, label)
    image = transform_image_only(image)

    return image, label


def transform_image_only(image):
    if random.random() > 0.5:
        # random change the colors of the picture
        image = RandomChoice([Grayscale(num_output_channels=3), ColorJitter(1, 1, 1)])(image)
    return image


def transform_both(image, label):
    # Random horizontal flipping
    if random.random() > 0.5:
        image = hflip(image)
        label = hflip(label)

    # Random vertical flipping
    if random.random() > 0.5:
        image = vflip(image)
        label = vflip(label)

    # Random rotation
    if random.random() > 0.5:
        angle = random.randint(0, 90)
        image = rotate(image, angle)
        label = rotate(label, angle)

    return image, label


augmentation_func = transform
