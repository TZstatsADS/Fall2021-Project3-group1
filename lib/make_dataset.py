"""
Filename: make_dataset.py
Scripts to download or generate data.
"""

import os

import numpy as np
import pandas as pd
from PIL import Image


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

    :return ((pd.DataFrame)):
    """
    # TODO: MAKE SURE PARAMS ARE CORRECT
    # Set global params
    DATA_PATH = "../data/"
    NUM_CLEAN_IMAGES = 10000

    # Get list of noisy, clean image files
    # TODO: Make this more readable
    img_files = sorted(os.listdir(DATA_PATH + "images/"))
    img_files = list(map(lambda x: DATA_PATH + "images/" + x, img_files))
    clean_img_files = img_files[:NUM_CLEAN_IMAGES]
    noisy_img_files = img_files[NUM_CLEAN_IMAGES:]

    # Get dataframes of noisy, clean images
    clean_img_df = get_df_from_img_file_lst(clean_img_files)
    noisy_img_df = get_df_from_img_file_lst(noisy_img_files)

    # Read labels
    clean_labels = pd.read_csv(DATA_PATH + "clean_labels.csv")
    noisy_labels = pd.read_csv(DATA_PATH + "noisy_labels.csv")

    return clean_img_df, clean_labels, noisy_img_df, noisy_labels
