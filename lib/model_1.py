import os

import numpy as np
import tensorflow as tf

from build_features import build_features, get_rgb_histogram, get_oriented_gradients_histogram, get_texture
from make_dataset import read_data


def sequential_ann():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation="relu", kernel_initializer="he_normal", input_shape=(352,)),
        tf.keras.layers.BatchNormalization(momentum=0.9),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")]
    )
    return model


def main():
    if "feature_mtx.txt" in os.listdir("../output/"):
        print("Loading processed data...")
        feature_mtx = np.loadtxt("../output/feature_mtx.txt")
        clean_labels = np.genfromtxt('../data/clean_labels.csv', delimiter=',', dtype="int8")
        noisy_labels = np.genfromtxt('../data/noisy_labels.csv', delimiter=',', dtype="int8")
    else:
        imgs, clean_labels, noisy_labels = read_data()
        feature_mtx = build_features(imgs, [get_rgb_histogram,
                                            get_oriented_gradients_histogram,
                                            get_texture])
        np.savetxt("../output/feature_mtx.txt", feature_mtx, fmt="%1.9f")

    # Split to train/test
    n_validation = 2000
    train_x, train_y = feature_mtx[n_validation:], noisy_labels[n_validation:]
    test_x, test_y = feature_mtx[:n_validation], clean_labels[:n_validation]

    print("Training Shapes: {}, {}".format(train_x.shape, train_y.shape))
    print("Testing Shapes: {}, {}".format(test_x.shape, test_y.shape))
    # Build model
    model = sequential_ann()
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(train_x, train_y, epochs=20)
    print(model.evaluate(test_x, test_y))


if __name__ == "__main__":
    main()
