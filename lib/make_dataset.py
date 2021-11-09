"""
Filename: make_dataset.py
Scripts to download or generate data.
"""

import os
import cv2

import numpy as np


def get_arr_from_img_file(img_path):
    """ Returns 3D array containing RGB values of an image.
    :param (str) img_path: path to image
    :return (numpy.array): dimension (n, k, 3) if original image is n * k.
    """
    img = Image.open(img_path)
    return np.asarray(img).flatten()


# TODO: Try parallelizing this function
def get_df_from_img_file_lst(img_files):
    """Given list of paths to images, function reads/vectorizes each image,
    stores as pandas.DataFrame. Rows are images and columns are RGB values.
    :param (list) img_files:
    :return (pandas.DataFrame)
    """
    images = list(map(get_arr_from_img_file, img_files))
    return pd.DataFrame(images).transpose()


def read_data():
    """
    Reads images, labels. Returns four pandas.DataFrame objects corresponding
    to vectorize clean images, clean labels, noisy images, noisy labels.

    :return (array-like, array-like, array-like):
    """
    print("Reading data...")
    n_img, n_noisy = 50000, 40000
    n_clean_noisy = n_img - n_noisy
    imgs = np.empty((n_img, 32, 32, 3))
    for i in range(n_img):
        img_fn = f'../data/images/{i + 1:05d}.png'
        imgs[i, :, :, :] = cv2.cvtColor(cv2.imread(img_fn), cv2.COLOR_BGR2RGB)
    clean_labels = np.genfromtxt('../data/clean_labels.csv', delimiter=',', dtype="int8")
    noisy_labels = np.genfromtxt('../data/noisy_labels.csv', delimiter=',', dtype="int8")

    return imgs, clean_labels, noisy_labels
