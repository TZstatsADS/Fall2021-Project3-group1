import numpy as np
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


def main():
    """TESTING CELL"""
    img_path = "../data/images/00001.png"
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    rgb_hist = get_rgb_histogram(img)
    print(rgb_hist)
    print(rgb_hist.shape)



if __name__ == "__main__":
    main()
