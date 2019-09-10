import numpy as np

# from scipy.misc import imsave, imread
from imageio import imsave, imread

from mrf import denoise_image

SOURCE_SIMILARITY_FACTOR = 7
NEIGHBORS_SIMILARITY_FACTOR = 10


def _get_bg_color(colors, counts):
    return colors[np.argmax(counts)]


def denoise_colored_image(img):
    colors, counts = np.unique(img, return_counts=True)

    print("There are %d colors in the given image of shape: %s" % (len(colors), str(img.shape)))

    bg_color = _get_bg_color(colors, counts)

    for color in colors:
        if color == bg_color:
            continue
        print("Working on color %s" % color)
        bw_img = np.where(img == color, 1, -1)
        denoised_image = denoise_image(bw_img, SOURCE_SIMILARITY_FACTOR, NEIGHBORS_SIMILARITY_FACTOR)

        # if a pixel from the segment was "cleaned" from the image after the MRF, we convert it to background color.
        # otherwise, if a pixel from outside the segment has joined the segment after the MRF, we convert it to color.
        img = np.where((denoised_image == -1) & (img == color),
                       bg_color,
                       np.where((denoised_image == 1) & (img != color),
                                color,
                                img))
    colors, counts = np.unique(img, return_counts=True)
    newColors = list(range(len(colors)))
    countSortedInd = np.argsort(-counts)  # get sorted indices from big to small
    colors = colors[countSortedInd]
    newImg = np.zeros(img.shape)
    for i in range(len(colors)):
        newImg[np.where(img == colors[i])] = newColors[i]
    return newImg


def main():
    image_name = "1seg"
    img = imread(image_name + ".jpg")

    img = denoise_colored_image(img)
    img = np.asarray(img, np.int32)
    imsave("%s-clean.jpg" % image_name, img)


if __name__ == '__main__':
    main()
