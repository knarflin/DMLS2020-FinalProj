#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import time

import crypten
import torch
from examples.meters import AverageMeter

import os
import cv2
import numpy as np
import pdb
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


from torchvision import transforms


class DataSet:
    def __init__(self):
        self._path = ""
        self._max_h = None
        self._max_w = None
        self._stddev_h = None
        self._stddev_w = None
        self._num_of_data = None
        self._mean_h = None
        self._mean_w = None

    @property
    def path(self):
        return self._path

    @property
    def num_of_data(self):
        return self._num_of_data

    def print_stats(self):

        print("Dataset Path:", self._path)
        print("Num of data:", self._num_of_data)
        print("Mean Height, Width:", self._mean_h, self._mean_w)
        print("Stddev. Height, Width:", self._stddev_h, self._stddev_w)
        print("Max Height, Width:", self._max_h, self._max_w)


class TrainingDataSet(DataSet):
    def __init__(self, dataset_path):
        DataSet.__init__(self)
        self.fetch_data_stats(dataset_path)

    def fetch_data_stats(self, train_dir):

        class_dirs = sorted(os.listdir(train_dir))
        total_metrics = None

        for class_dir in class_dirs:
            current_dir = os.path.join(train_dir, class_dir)
            class_id = int(class_dir)

            image_files = os.listdir(current_dir)

            metrics = np.zeros((len(image_files), 2), dtype=int)

            for i, image_file in enumerate(image_files, start=0):
                image_path = os.path.join(current_dir, image_file)
                img = cv2.imread(image_path)

                metrics[i, 0] = img.shape[0]
                metrics[i, 1] = img.shape[1]

            if not isinstance(total_metrics, np.ndarray):
                total_metrics = metrics
            else:
                total_metrics = np.concatenate(
                    (total_metrics, metrics), axis=0)

        self._path = train_dir
        self._num_of_data = total_metrics.shape[0]
        self._mean_h, self._mean_w = np.mean(total_metrics, axis=0)
        self._stddev_h, self._stddev_w = np.std(total_metrics, axis=0)
        self._max_h, self._max_w = np.amax(total_metrics, axis=0)


def train_linear_svm(features, labels, epochs=50, lr=0.5, print_time=False):
    # Initialize random weights
    w = features.new(torch.randn(1, features.size(0)))
    b = features.new(torch.randn(1))

    if print_time:
        pt_time = AverageMeter()
        end = time.time()

    for epoch in range(epochs):
        # Forward
        label_predictions = w.matmul(features).add(b).sign()

        # Compute accuracy
        correct = label_predictions.mul(labels)
        accuracy = correct.add(1).div(2).mean()
        if crypten.is_encrypted_tensor(accuracy):
            accuracy = accuracy.get_plain_text()

        # Print Accuracy once
        if crypten.communicator.get().get_rank() == 0:
            logging.info(
                f"Epoch {epoch} --- Training Accuracy %.2f%%" % (
                    accuracy.item() * 100)
            )

        # Backward
        loss_grad = -labels * (1 - correct) * 0.5  # Hinge loss
        b_grad = loss_grad.mean()
        w_grad = loss_grad.matmul(features.t()).div(loss_grad.size(1))

        # Update
        w -= w_grad * lr
        b -= b_grad * lr

        if print_time:
            iter_time = time.time() - end
            pt_time.add(iter_time)
            logging.info("    Time %.6f (%.6f)" % (iter_time, pt_time.value()))
            end = time.time()

    return w, b


def evaluate_linear_svm(features, labels, w, b):
    """Compute accuracy on a test set"""
    predictions = w.matmul(features).add(b).sign()
    correct = predictions.mul(labels)
    accuracy = correct.add(1).div(2).mean().get_plain_text()
    if crypten.communicator.get().get_rank() == 0:
        print("Test accuracy %.2f%%" % (accuracy.item() * 100))


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.cnn1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.cnn2 = nn.Conv2d(64, 128, 3, 1, 1)  # [128, 24, 24]
        self.cnn3 = nn.Conv2d(128, 256, 3, 1, 1)  # [256, 12, 12]

        self.fc1 = nn.Linear(256*6*6, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 43)

        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.ReLU(),
        )

    def forward(self, x):

        # Conv Layers
        out = self.cnn1(x)  # [64, 48, 48]
        # out = nn.BatchNorm2d(64)
        out = F.relu(out)
        out = F.max_pool2d(out, 2)  # [64, 24, 24]

        out = self.cnn2(out)  # [128, 24, 24]
        # nn.BatchNorm2d(128)
        out = F.relu(out)
        out = F.max_pool2d(out, 2)  # [128, 12, 12]

        out = self.cnn3(out)  # [256, 12, 12]
        # nn.BatchNorm2d(256)
        out = F.relu(out)
        out = F.max_pool2d(out, 2)  # [256, 6, 6]

        # out = out.view(out.size()[0], -1)  # Flattern
        out = out.view(-1, 256 * 6 * 6)

        # # Fully Connected Layers
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        return out


'''
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 48, 48]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [64, 24, 24]

            nn.Conv2d(64, 128, 3, 1, 1),  # [128, 24, 24]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 12, 12]

            nn.Conv2d(128, 256, 3, 1, 1),  # [256, 12, 12]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),  # [256, 12, 12]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [256, 6, 6]
        )
        self.fc = nn.Sequential(
            # nn.Linear(512*4*4, 1024),
            nn.Linear(256*6*6, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 43)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)  # Flattern
        return self.fc(out)
'''


def run_mpc_cnn(
    epochs=50, examples=50, features=100, lr=0.001, skip_plaintext=False
):
    crypten.init()

    # Set random seed for reproducibility
    torch.manual_seed(1)  # TODO

    rank = crypten.communicator.get().get_rank()

    training_dataset_path = "/data/train/site%d" % rank
    validation_dataset_path = "/data/validation"

    training_dataset = TrainingDataSet(training_dataset_path)
    validation_dataset = TrainingDataSet(validation_dataset_path)

    training_dataset.print_stats()
    validation_dataset.print_stats()

    do_train(training_dataset, validation_dataset)
    # TODO
    return


def do_train(training_datastats, validation_datastats):

    from torchvision.datasets import ImageFolder

    custom_transform = transforms.Compose(
        [transforms.Resize([48, 48]),
         transforms.ToTensor()]
    )
    train_set = ImageFolder(training_datastats.path, custom_transform)
    valid_set = ImageFolder(validation_datastats.path, custom_transform)

    num_of_train_set = len(train_set)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=32, shuffle=False)

    # crypten.save(features_alice, filenames["features_alice"], src=ALICE)

    dummy_input = torch.randn(1, 3, 48, 48)

    model_plaintext = Classifier()
    model = crypten.nn.from_pytorch(model_plaintext, dummy_input)
    model.encrypt()

    # loss = crypten.nn.CrossEntropyLoss()
    loss = crypten.nn.MSELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    num_epoch = 100

    rank = crypten.communicator.get().get_rank()
    print("My rank is", rank)  # debug point

    # label_eye = torch.eye(2)
    for epoch in range(num_epoch):
        model.train()
        for i, data in enumerate(train_loader):
            # print("Rank %d, i = %d" % (rank, i))

            site0_feature_enc = crypten.cryptensor(data[0], src=0)
            site1_feature_enc = crypten.cryptensor(data[0], src=1)
            site2_feature_enc = crypten.cryptensor(data[0], src=2)
            site3_feature_enc = crypten.cryptensor(data[0], src=3)

            print(data[0].shape)
            x_combined_enc = crypten.cat(
                [site0_feature_enc, site1_feature_enc, site2_feature_enc, site3_feature_enc], dim=0)

            x_combined_enc.requires_grad = True

            print(x_combined_enc.shape)

            site0_label_enc = crypten.cryptensor(data[1], src=0)
            site1_label_enc = crypten.cryptensor(data[1], src=1)
            site2_label_enc = crypten.cryptensor(data[1], src=2)
            site3_label_enc = crypten.cryptensor(data[1], src=3)

            # TODO: y_one_hot
            # y_one_hot = label_eye[y_encrypted[start:end]]
            # y_train = crypten.cryptensor(y_one_hot, requires_grad=True)

            y_combined_enc = crypten.cat(
                [site0_label_enc, site1_label_enc, site2_label_enc, site3_label_enc], dim=0)

            print(y_combined_enc.shape)

            # print(data)
            # optimizer.zero_grad()
            # train_pred = model(data[0].cuda())
            # batch_loss = loss(train_pred, data[1].cuda())

            train_pred = model(x_combined_enc)
            batch_loss = loss(train_pred, y_combined_enc)

            model.zero_grad()
            batch_loss.backward()
            model.update_parameters(learning_rate)

            # optimizer.step()

        # compute accuracy every epoch
        pred = train_pred.get_plain_text().argmax(1)
        correct = pred.eq(y_combined_enc)
        correct_count = correct.sum(0, keepdim=True).float()
        accuracy = correct_count.mul_(100.0 / train_pred.size(0))

        loss_plaintext = batch_loss.get_plain_text().item()
        print(
            f"Rank {rank} Epoch {epoch} completed: "
            f"Loss {loss_plaintext:.4f} Accuracy {accuracy.item():.2f}"
        )

        # train_loss = 0.0
        # train_acc = 0.0
        # model.eval()
        # for i, data in enumerate(train_loader):
        #     train_pred = model(data[0].cuda())
        #     batch_loss = loss(train_pred, data[1].cuda())
        #     train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(),
        #                                   axis=1) == data[1].numpy())
        #     train_loss += batch_loss.item()
        # print('Train Acc: %3.6f Loss: %3.6f' %
        #       (train_acc/len(train_set), train_loss/len(train_set)))

        if rank == 0:
            val_acc = 0.0
            model.eval()
            for i, data in enumerate(valid_loader):
                val_pred = model(data[0])
                val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(),
                                            axis=1) == data[1].numpy())

            print('Val Acc: %3.6f' % (val_acc/len(valid_set)))


def run_mpc_linear_svm(
    epochs=50, examples=50, features=100, lr=0.5, skip_plaintext=False
):
    crypten.init()

    # Set random seed for reproducibility
    torch.manual_seed(1)

    # Initialize x, y, w, b
    x = torch.randn(features, examples)
    w_true = torch.randn(1, features)
    b_true = torch.randn(1)
    y = w_true.matmul(x) + b_true
    y = y.sign()

    if not skip_plaintext:
        logging.info("==================")
        logging.info("PyTorch Training")
        logging.info("==================")
        w_torch, b_torch = train_linear_svm(
            x, y, epochs=epochs, lr=lr, print_time=True)

    # Encrypt features / labels
    x = crypten.cryptensor(x)
    y = crypten.cryptensor(y)

    logging.info("==================")
    logging.info("CrypTen Training")
    logging.info("==================")
    w, b = train_linear_svm(x, y, epochs=epochs, lr=lr, print_time=True)

    if not skip_plaintext:
        logging.info("PyTorch Weights  :")
        logging.info(w_torch)
    logging.info("CrypTen Weights:")
    logging.info(w.get_plain_text())

    if not skip_plaintext:
        logging.info("PyTorch Bias  :")
        logging.info(b_torch)
    logging.info("CrypTen Bias:")
    logging.info(b.get_plain_text())
