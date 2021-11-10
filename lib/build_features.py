import timeit

import cv2
import mahotas
import numpy as np
from mahotas.features import haralick
from skimage.feature import hog


def get_rgb_histogram(img, no_bins=5):
    """Returns array of RGB histogram.

    :param (array-like) img : vectorized image
    :param (int) no_bins : number of bins for hist for each color

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


def get_texture(img):
    """Returns array of texture feature.

    :param (array-like) img: vectorized image

    :return (array-like): feature vector of shape (13, )
    """
    # Setting image to grayscale, then filtering using Gaussian filter
    img_filtered = img[:, :, 0]
    img_filtered = mahotas.gaussian_filter(img_filtered, 4)
    thresh = (img_filtered > img_filtered.mean())
    labeled, n = mahotas.label(thresh)

    feature_texture = mahotas.features.haralick(labeled).mean(axis=0)
    return feature_texture


def build_features(imgs_mtx, feature_fns):
    print("Building features...")
    feature_mtx = list()
    for img in imgs_mtx:
        features = [f(img) for f in feature_fns]
        features_concat = np.concatenate(features, axis=None)
        feature_mtx.append(features_concat)
    return np.array(feature_mtx)


def main():
    # Collect data
    print("Collecting data")
    collection_start = timeit.timeit()
    n_img = 50000
    imgs = np.empty((n_img, 32, 32, 3))
    for i in range(n_img):
        img_fn = f'../data/images/{i + 1:05d}.png'
        imgs[i, :, :, :] = cv2.cvtColor(cv2.imread(img_fn), cv2.COLOR_BGR2RGB)
    collection_end = timeit.timeit()

    # Build features
    print("Building features")
    features_start = timeit.timeit()
    feature_mtx = build_features(imgs, [get_rgb_histogram,
                                        get_oriented_gradients_histogram,
                                        get_texture])
    features_end = timeit.timeit()

    # Print summary stats
    print("Original matrix shape: {}".format(imgs.shape))
    print("Processed matrix shape: {}".format(feature_mtx.shape))

    print("Time Elapsed (Data collection): {}".format(collection_end - collection_start))
    print("Time Elapsed (Feature Engineering): {}".format(features_end - features_start))


if __name__ == "__main__":
    main()
