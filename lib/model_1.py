import os
import time
import numpy as np
import tensorflow as tf
import csv
from tensorflow.keras.layers import Dense, BatchNormalization, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.regularizers import l2

from build_features import build_features, get_rgb_histogram, get_oriented_gradients_histogram, get_texture
from make_dataset import read_data


def sequential_ann():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation="relu", kernel_initializer="he_normal", input_shape=(352,)),
        tf.keras.layers.BatchNormalization(momentum=0.9),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.BatchNormalization(momentum=0.6),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")]
    )
    return model


def cnn(weight_decay):
    model = tf.keras.models.Sequential([
        # Adding artificial noise by flipping, rotating, changing contrasts
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),

        Conv2D(32, (3, 3), padding="same", activation="relu", kernel_regularizer=l2(weight_decay),
               input_shape=(32, 32, 3)),
        BatchNormalization(),
        Conv2D(32, (3, 3), padding="same", activation="relu", kernel_regularizer=l2(weight_decay)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),

        Conv2D(64, (3, 3), padding="same", activation="relu", kernel_regularizer=l2(weight_decay)),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding="same", activation="relu", kernel_regularizer=l2(weight_decay)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),

        Conv2D(128, (3, 3), padding="same", activation="relu", kernel_regularizer=l2(weight_decay)),
        BatchNormalization(),
        Conv2D(128, (3, 3), padding="same", activation="relu", kernel_regularizer=l2(weight_decay)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.4),

        Flatten(),
        Dense(10, activation="softmax")
    ])
    return model


def run_cnn_clean(n_validation, epochs, batch_size, weight_decay, learning_rate):
    """Running CNN model only on clean data. Useful to test if model architecture is good."""
    imgs, clean_labels, noisy_labels = read_data()
    imgs = imgs / 255.0

    # Split into train/test
    train_x, train_y = imgs[n_validation:10000], clean_labels[n_validation:]
    test_x, test_y = imgs[:n_validation], clean_labels[:n_validation]

    print("Training Shapes: {}, {}".format(train_x.shape, train_y.shape))
    print("Testing Shapes: {}, {}".format(test_x.shape, test_y.shape))
    # Build model
    model = cnn(weight_decay)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss="sparse_categorical_crossentropy",
                  metrics=["sparse_categorical_accuracy"])

    start_time = time.time()
    hist = model.fit(x=train_x,
                     y=train_y,
                     epochs=epochs,
                     batch_size=batch_size,
                     validation_data=(test_x, test_y))
    end_time = time.time()
    return hist.history, end_time - start_time


def tune_cnn_params_m1_clean():
    epochs = [10, 20, 25]
    validation_szs = [1000, 1500]
    learning_rates = [1e-3, 1e-4]
    batch_size = 64
    weight_decay = 1e-4

    res_log = [["Model",
                "# Epochs",
                "Validation Size",
                "Learning Rate",
                "Weight Decay",
                "Batch Size",
                "Validation Loss", "Validation Accuracy", "Training Time"]]
    for epoch in epochs:
        for n_validation in validation_szs:
            for lr in learning_rates:
                hist, time = run_cnn_clean(n_validation, epoch, batch_size, weight_decay, lr)
                res = ["CNN", epoch, n_validation, lr, weight_decay, batch_size,
                       hist["val_loss"][-1], hist["val_sparse_categorical_accuracy"][-1], time]
                res_log.append(res)
                print(res)

    # Write output to disk
    file = open("../output/cnn_clean_tests.csv", "w+", newline="")
    with file:
        write = csv.writer(file)
        write.writerows(res_log)


def run_sequential_ann():
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
    model.fit(train_x, train_y, epochs=20, batch_size=256, shuffle=True)
    print(model.evaluate(test_x, test_y))


if __name__ == "__main__":
    tune_cnn_params_m1_clean()
