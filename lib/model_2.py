import csv
import numpy as np
import numpy as np
import os
import pickle
import tensorflow as tf
import tensorflow as tf
import time
import time
from build_features import build_features, get_rgb_histogram, get_oriented_gradients_histogram, get_texture
from make_dataset import read_data
from make_dataset import read_data
from model_1 import model_1
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, BatchNormalization, Conv2D, MaxPooling2D, Dropout, Flatten, Concatenate
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical


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
    b1 = Dense(128, activation='linear')(b1)
    b1 = Model(inputs=img_input, outputs=b1)

    # Second branch
    b2 = Dense(128, activation="linear")(label_input)
    b2 = Model(inputs=label_input, outputs=b2)

    combined = Concatenate()([b1.output, b2.output])
    b_combined = Dense(64, activation="relu")(combined)
    b_combined = Dropout(0.2)(b_combined)
    b_combined = Dense(n_classes, activation="softmax")(b_combined)

    model = Model(inputs=[b1.input, b2.input], outputs=b_combined)
    return model


def train_model_2():
    # Read data, conduct basic standardization procedures
    imgs, clean_labels, noisy_labels = read_data()
    imgs = imgs / 255.0
    clean_labels = to_categorical(np.concatenate(clean_labels, axis=None))
    noisy_labels = to_categorical(np.concatenate(noisy_labels, axis=None))

    if "label_classifier" in os.listdir("../output/"):
        print("Loading label classifier...")
        l_classifier = tf.keras.models.load_model("../output/label_classifier")
    else:
        train_label_classifier()
        l_classifier = tf.keras.models.load_model("../output/label_classifier")

    # Predict labels for 40000 noisy labels
    predicted_noisy_labels = l_classifier.predict([imgs[10000:], noisy_labels[10000:]])
    predicted_noisy_labels = np.argmax(predicted_noisy_labels, axis=1)
    predicted_noisy_labels = to_categorical(np.concatenate(predicted_noisy_labels, axis=None))
    target = np.concatenate([clean_labels, predicted_noisy_labels])

    # Train model
    start = time.time()
    print("Training image classifier...")
    train_x, train_y = imgs[:], target[:]
    val_x, val_y = imgs[:2000], target[:2000]
    classifier = model_1(10)
    classifier.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(),
                       metrics=['accuracy'])
    res = classifier.fit(train_x, train_y, batch_size=256, epochs=10, verbose=1,
                         validation_data=(val_x, val_y))
    hist = res.history
    end = time.time()
    hist["time (s)"] = end - start

    # Save model, training history
    classifier.save("../output/model_2")
    with open("../output/model_2_train_history", "wb") as training_hist:
        pickle.dump(hist, training_hist)


def train_label_classifier():
    # Read data, conduct basic standardization procedures
    imgs, clean_labels, noisy_labels = read_data()
    imgs = imgs / 255.0
    clean_labels = to_categorical(np.concatenate(clean_labels, axis=None))
    noisy_labels = to_categorical(np.concatenate(noisy_labels, axis=None))

    # Train label classifier
    start = time.time()
    print("Training label classifier...")
    l_classifier = label_classifier(10)
    l_classifier.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(),
                         metrics=['accuracy'])
    res = l_classifier.fit(x=[imgs[:10000], noisy_labels[:10000]], y=clean_labels[:10000],
                           validation_data=([imgs[:2000], noisy_labels[:2000]], clean_labels[:2000]),
                           epochs=15,
                           batch_size=256)
    hist = res.history
    end = time.time()
    hist["time (s)"] = end - start

    # Save model, training history
    l_classifier.save("../output/label_classifier")
    with open("../output/label_classifier_train_history", "wb") as training_hist:
        pickle.dump(hist, training_hist)
