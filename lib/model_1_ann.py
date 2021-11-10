from __future__ import print_function

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD


class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(15, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        output = self.linear_relu_stack(x)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target,
                                    reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1,
                                 keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


def next_batch(inputs, targets, batchSize):
    # loop over the dataset
    for i in range(0, inputs.shape[0], batchSize):
        # yield a tuple of the current batched data and labels
        yield inputs[i:i + batchSize], targets[i:i + batchSize]


def main():
    # load the images
    n_img = 50000
    n_noisy = 40000
    n_clean_noisy = n_img - n_noisy
    imgs = np.empty((n_img, 32, 32, 3))
    for i in range(n_img):
        img_fn = f'../data/images/{i + 1:05d}.png'
        imgs[i, :, :, :] = cv2.cvtColor(cv2.imread(img_fn), cv2.COLOR_BGR2RGB)

    # load the labels
    clean_labels = np.genfromtxt('../data/clean_labels.csv', delimiter=',',
                                 dtype="int8")
    noisy_labels = np.genfromtxt('../data/noisy_labels.csv', delimiter=',',
                                 dtype="int8")

    """Feature extraction"""
    # RGB histogram dataset construction
    no_bins = 6
    bins = np.linspace(0, 255, no_bins)  # the range of the rgb histogram
    target_vec = np.empty(n_img)
    feature_mtx = np.empty((n_img, 3 * (len(bins) - 1)))
    i = 0
    for i in range(n_img):
        # The target vector consists of noisy labels
        target_vec[i] = noisy_labels[i]

        # Use the numbers of pixels in each bin for all three channels as the features
        feature1 = np.histogram(imgs[i][:, :, 0], bins=bins)[0] / (32 * 32)
        feature2 = np.histogram(imgs[i][:, :, 1], bins=bins)[0] / (32 * 32)
        feature3 = np.histogram(imgs[i][:, :, 2], bins=bins)[0] / (32 * 32)

        # Concatenate three features
        feature_mtx[i,] = np.concatenate((feature1, feature2, feature3),
                                         axis=None)
        i += 1
    print("Feature Extraction Complete")

    """Split train/test."""
    validation_sz = 1000
    validation_x, validation_y = feature_mtx[:validation_sz], target_vec[
                                                              :validation_sz]
    training_x, training_y = feature_mtx[validation_sz:], target_vec[
                                                          validation_sz:]
    print("Beggining Training.")
    """TESTING PYTORCH SOLUTION"""
    BATCH_SIZE = 100
    EPOCHS = 490
    LR = 1e-3

    trainX, trainY = torch.from_numpy(training_x).float(), torch.from_numpy(
        training_y).float()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Model1().to(DEVICE)
    opt = SGD(model.parameters(), lr=LR)
    lossFunc = nn.CrossEntropyLoss()

    # training loop
    for epoch in range(0, EPOCHS):
        print("[INFO] epoch: {}...".format(epoch + 1))
        trainLoss = 0
        trainAcc = 0
        samples = 0
        model.train()

        for (batchX, batchY) in next_batch(trainX, trainY, BATCH_SIZE):
            # flash data to the current device, run it through our
            # model, and calculate loss
            (batchX, batchY) = (batchX.to(DEVICE), batchY.to(DEVICE))
            predictions = model(batchX)
            loss = lossFunc(predictions, batchY.long())
            # zero the gradients accumulated from the previous steps,
            # perform backpropagation, and update model parameters
            opt.zero_grad()
            loss.backward()
            opt.step()
            # update training loss, accuracy, and the number of samples
            # visited
            trainLoss += loss.item() * batchY.size(0)
            trainAcc += (predictions.max(1)[1] == batchY).sum().item()
            samples += batchY.size(0)
        trainTemplate = "epoch: {} train loss: {:.3f} train accuracy: {:.3f}"
        print(trainTemplate.format(epoch + 1, (trainLoss / samples),
                                   (trainAcc / samples)))


if __name__ == "__main__":
    main()
