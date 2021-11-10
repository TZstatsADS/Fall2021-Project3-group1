import csv
import numpy as np
import os
import pickle
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
