import csv
import numpy as np
import os
import tensorflow as tf
import time
from build_features import build_features, get_rgb_histogram, get_oriented_gradients_histogram, get_texture
from make_dataset import read_data
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, BatchNormalization, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
import pickle


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
    epochs = [5]
    validation_szs = [2000]
    learning_rates = [1e-3]
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


def model_1(num_classes):
    model = Sequential([
        # CNNs w/ ReLu and MaxPooling act as feature extractor
        Conv2D(32, kernel_size=(3, 3), activation='linear', padding='same', input_shape=(32, 32, 3)),
        LeakyReLU(alpha=0.1),
        MaxPooling2D((2, 2), padding='same'),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='linear', padding='same'),
        LeakyReLU(alpha=0.1),
        MaxPooling2D(pool_size=(2, 2), padding='same'),
        Dropout(0.25),
        Conv2D(128, (3, 3), activation='linear', padding='same'),
        LeakyReLU(alpha=0.1),
        MaxPooling2D(pool_size=(2, 2), padding='same'),
        Dropout(0.4),

        Flatten(),
        Dense(128, activation='linear'),
        LeakyReLU(alpha=0.1),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    return model


def train_model_1():
    # Read data
    imgs, clean_labels, noisy_labels = read_data()
    imgs = imgs / 255.0

    # Split into training/validation
    start = time.time()
    train_X, train_Y_one_hot = imgs[:], to_categorical(np.concatenate((noisy_labels[:]), axis=None))
    test_X, test_Y_one_hot = imgs[:2000], to_categorical(clean_labels[:2000])

    # Set parameters
    batch_size = 64
    epochs = 15
    num_classes = 10

    # Train model
    model = model_1(num_classes)
    model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
    training_res = model.fit(train_X, train_Y_one_hot, batch_size=batch_size, epochs=epochs, verbose=1,
                             validation_data=(test_X, test_Y_one_hot))
    end = time.time()
    hist = training_res.history
    hist["time (s)"] = end - start

    # Save model, training history
    model.save("../output/model_1")
    with open("../output/model_1_train_history", "wb") as training_hist:
        pickle.dump(hist, training_hist)


if __name__ == "__main__":
    train_model_1()
