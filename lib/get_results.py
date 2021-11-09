from model_1 import cnn
from make_dataset import read_data
import time
import csv
import tensorflow as tf


def run_cnn(n_validation, epochs, batch_size, weight_decay, learning_rate):
    """Running CNN model. Useful to test if model architecture is good."""
    imgs, clean_labels, noisy_labels = read_data()
    imgs = imgs / 255.0

    # Split into train/test
    train_x, train_y = imgs[n_validation:], noisy_labels[n_validation:]
    test_x, test_y = imgs[:n_validation], clean_labels[:n_validation]

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


def get_model1_res():
    """Recording Model I results w/ varying parameters."""
    epochs = [10, 20, 25]
    validation_szs = [1000, 1500]
    learning_rates = [1e-3, 1e-4]
    batch_size = 256
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
                hist, time = run_cnn(n_validation, epoch, batch_size, weight_decay, lr)
                res = ["CNN", epoch, n_validation, lr, weight_decay, batch_size,
                       hist["val_loss"][-1], hist["val_sparse_categorical_accuracy"][-1], time]
                res_log.append(res)
                print(res)

    # Write output to disk
    file = open("../output/cnn_model1_tests.csv", "w+", newline="")
    with file:
        write = csv.writer(file)
        write.writerows(res_log)


if __name__ == "__main__":
    get_model1_res()
