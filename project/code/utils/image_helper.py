import numpy as np


def get_image_mask_corners_pixels(image, pixel_for_mask):
    coords = list()

    image_mask_pixels_coords = np.where(image == pixel_for_mask)
    top_x = np.argmax(image_mask_pixels_coords[0])
    bottom_x = np.argmin(image_mask_pixels_coords[0])

    top_y = np.argmax(image_mask_pixels_coords[1])
    bottom_y = np.argmin(image_mask_pixels_coords[1])

    coords.append((image_mask_pixels_coords[0][top_x], image_mask_pixels_coords[1][top_x]))
    coords.append((image_mask_pixels_coords[0][bottom_x], image_mask_pixels_coords[1][bottom_x]))
    coords.append((image_mask_pixels_coords[0][top_y], image_mask_pixels_coords[1][top_y]))
    coords.append((image_mask_pixels_coords[0][bottom_y], image_mask_pixels_coords[1][bottom_y]))

    return coords


def get_image_center_pixel(image, pixel_for_mask):
    points = np.argwhere(image == pixel_for_mask)
    center = points.mean(axis=0)
    return center.astype(int)
