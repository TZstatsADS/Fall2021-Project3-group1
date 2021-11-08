import numpy as np
from skimage.feature import hog
import cv2


def get_rgb_histogram(img, no_bins=5):
    """ Returns array of RGB histogram.

    :param img (array-like): vectorized image
    :param no_bins (int): number of bins for hist for each color

    :return (array-like): array of shape (no_bins * 3, )
    """
    bins = np.linspace(0, 255, no_bins + 1)
    feature_r = np.histogram(img[:, :, 0], bins=bins)[0]
    feature_g = np.histogram(img[:, :, 1], bins=bins)[0]
    feature_b = np.histogram(img[:, :, 2], bins=bins)[0]

    return np.concatenate((feature_r, feature_g, feature_b), axis=None)


def get_oriented_gradients_histogram(img, no_orientations=9):
    """Wrapper around skimage.feature.hog function.

    :param (array-like) img: vectorized image
    :param (int) no_orientations: number of orientation bins

    :return (array-like): feature vector using (3, 3) blocks,
    L2 norm and normalizing image before processing.
    """
    feature_hog = hog(img,
                      orientations=no_orientations,
                      cells_per_block=(3, 3),
                      feature_vector=True,
                      transform_sqrt=True,
                      block_norm="L2",
                      multichannel=True)
    return feature_hog


def main():
    """TESTING CELL"""
    img_path = "../data/images/00001.png"
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    # Testing RGB histogram
    # rgb_hist = get_rgb_histogram(img)
    # print(rgb_hist)
    # print(rgb_hist.shape)

    # Testing HOG
    hog = get_oriented_gradients_histogram(img, 9)
    print(hog)
    print(hog.shape)


if __name__ == "__main__":
    main()
