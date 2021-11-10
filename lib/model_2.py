import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
import numpy as np
from make_dataset import read_data


def base_cnn():
    model = VGG16()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    return model


def label_cleaning_model():
    ...


def image_classification_model():
    ...


def model_2():
    imgs, clean_labels, noisy_labels = read_data()
    imgs = imgs / 255.0

    # Pass all images 1st through a CNN acting as a feature vectorizer
    base_cnn = base_cnn()
    features_mtx = []
    for img in imgs:
        img_processed = base_cnn.predict(img)
        features_mtx.append(img_processed)
    features_mtx = np.array(features_mtx)
    ...
