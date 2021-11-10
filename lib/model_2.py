import tensorflow as tf
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
import numpy as np
from make_dataset import read_data
import csv
import numpy as np
import os
import tensorflow as tf
import time
from build_features import build_features, get_rgb_histogram, get_oriented_gradients_histogram, get_texture
from make_dataset import read_data
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, BatchNormalization, Conv2D, MaxPooling2D, Dropout, Flatten, Concatenate
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical


def base_cnn(input_shape):
    model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    return model


def label_classifier(n_classes):
    img_input = Input(shape=(32, 32, 3))
    label_input = Input(shape=(10,))

    # First branch is a CNN acting directly on image
    b1 = Conv2D(32, kernel_size=(3, 3), activation='linear', padding='same', input_shape=(32, 32, 3))(img_input)
    b1 = LeakyReLU(alpha=0.1)(b1)
    b1 = MaxPooling2D((2, 2), padding='same')(b1)
    b1 = Dropout(0.25)(b1)
    b1 = Conv2D(64, (3, 3), activation='linear', padding='same')(b1)
    b1 = LeakyReLU(alpha=0.1)(b1)
    b1 = MaxPooling2D(pool_size=(2, 2), padding='same')(b1)
    b1 = Dropout(0.25)(b1)
    b1 = Conv2D(128, (3, 3), activation='linear', padding='same')(b1)
    b1 = LeakyReLU(alpha=0.1)(b1)
    b1 = MaxPooling2D(pool_size=(2, 2), padding='same')(b1)
    b1 = Dropout(0.4)(b1)
    b1 = Flatten()(b1)
    b1 = Dense(64, activation='linear')(b1)
    b1 = Model(inputs=img_input, outputs=b1)

    # Second branch
    b2 = Dense(10, activation="linear")(label_input)
    b2 = Model(inputs=label_input, outputs=b2)

    combined = Concatenate()([b1.output, b2.output])
    b_combined = Dense(64, activation="relu")(combined)
    b_combined = Dense(n_classes, activation="softmax")(b_combined)

    model = Model(inputs=[b1.input, b2.input], outputs=b_combined)
    return model


    # model = Sequential([
    #     Flatten(),
    #     Dense(128, activation="relu"),
    #     Dense(n_classes, activation="softmax")
    # ])
    # return model


def validate_label_classifier():
    # Read data, conduct basic standardization procedures
    imgs, clean_labels, noisy_labels = read_data()
    imgs = imgs / 255.0
    clean_labels = to_categorical(np.concatenate(clean_labels, axis=None))
    noisy_labels = to_categorical(np.concatenate(noisy_labels, axis=None))

    # Using pre-trained MobileNet for feature extraction
    # print("Extracting features...")
    # feature_extractor = MobileNet(weights="imagenet", include_top=False, input_shape=(32, 32, 3))
    # imgs_processed = preprocess_input(imgs)
    # imgs_processed = feature_extractor.predict(imgs_processed)
    # imgs_processed = np.reshape(imgs_processed, (50000, 1024))


    # Train/test split
    print("Splitting data...")
    # imgs_with_noisy = np.concatenate((imgs_processed, noisy_labels), axis=1)
    # train_x, train_y = imgs_with_noisy[2000:10000], clean_labels[2000:10000]
    # val_x, val_y = imgs_with_noisy[:2000], clean_labels[:2000]
    model = label_classifier(10)
    model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
    model.fit(x=[imgs[2000:10000], noisy_labels[2000:10000]], y=clean_labels[2000:10000],
              validation_data=([imgs[:2000], noisy_labels[:2000]], clean_labels[:2000]),
              epochs=5,
              batch_size=256)

    # Build model params
    # print("Building model parameters")
    # input_shape = imgs_with_noisy[0].shape
    # n_classes = 10
    # batch_size = 64
    # epochs = 10
    #
    # # Train model
    # model = label_classifier(n_classes)
    # model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(),
    #               metrics=['accuracy'])
    # training_res = model.fit(train_x, train_y, batch_size=batch_size, epochs=500, verbose=1,
    #                          validation_data=(val_x, val_y))
    # val_res = model.evaluate(val_x, val_y)
    #
    # print("Validation loss: {}".format(val_res[0]))
    # print("Validation accuracy: {}".format(val_res[1]))


if __name__ == "__main__":
    validate_label_classifier()
