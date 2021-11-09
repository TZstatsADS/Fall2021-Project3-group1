"""
Filename: make_dataset.py
Scripts to download or generate data.
"""

import cv2

import numpy as np


def read_data():
    """
    Reads images, labels. Returns four Numpy arrays corresponding
    to vectorized images, clean labels, noisy labels.

    :return (array-like, array-like, array-like):
    """
    print("Reading data...")
    n_img, n_noisy = 50000, 40000
    imgs = np.empty((n_img, 32, 32, 3))
    for i in range(n_img):
        img_fn = f'../data/images/{i + 1:05d}.png'
        imgs[i, :, :, :] = cv2.cvtColor(cv2.imread(img_fn), cv2.COLOR_BGR2RGB)

    clean_labels = np.genfromtxt('../data/clean_labels.csv', delimiter=',', dtype="int8")
    noisy_labels = np.genfromtxt('../data/noisy_labels.csv', delimiter=',', dtype="int8")

    return imgs, clean_labels, noisy_labels


def get_train_val_split_m1(feature_mtx, clean_labels, noisy_labels, n_validation=2000):
    """Returns training/validation split.

    :param (array-like) feature_mtx: either matrix of images or processed features
    :param (array-like) clean_labels:
    :param (array-like) noisy_labels:
    :param (int) n_validation: number of values to keep for validation

    :return (array-like, array-like, array-like, array-like): train_x, train_y, test_x, test_y
    """
    train_x, train_y = feature_mtx[n_validation:], noisy_labels[n_validation:]
    test_x, test_y = feature_mtx[:n_validation], clean_labels[:n_validation]

    return train_x, train_y, test_x, test_y
