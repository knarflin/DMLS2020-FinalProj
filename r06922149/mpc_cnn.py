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


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.cnn1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(8*12*12, 43)

    def forward(self, x):

        # Conv Layers
        out = self.cnn1(x)  # [8, 48, 48]
        out = F.relu(out)
        out = F.max_pool2d(out, 4)  # [8, 12, 12]

        out = out.view(-1, 8 * 12 * 12)

        # Fully Connected Layers
        out = self.fc1(out)
        return out


def run_mpc_cnn():
    crypten.init()

    # Set random seed for reproducibility
    torch.manual_seed(1)  # TODO

    rank = crypten.communicator.get().get_rank()

    training_dataset_path = "./splitdata/train/site%d" % rank
    # training_dataset_path = "./splitdata/train/all"  # debug point
    validation_dataset_path = "./splitdata/validation"

    training_dataset = TrainingDataSet(training_dataset_path)
    validation_dataset = TrainingDataSet(validation_dataset_path)

    training_dataset.print_stats()
    validation_dataset.print_stats()

    do_train(training_dataset, validation_dataset)
    return


def do_train(training_datastats, validation_datastats, resize_hw=48):

    from torchvision.datasets import ImageFolder

    custom_transform = transforms.Compose(
        [transforms.Resize([resize_hw, resize_hw]),
         transforms.ToTensor()]
    )
    train_set = ImageFolder(training_datastats.path, custom_transform)
    valid_set = ImageFolder(validation_datastats.path, custom_transform)

    num_of_train_set = len(train_set)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=32, shuffle=False)

    # crypten.save(features_alice, filenames["features_alice"], src=ALICE)

    dummy_input = torch.randn(1, 3, resize_hw, resize_hw)

    model_plaintext = Classifier()
    model = crypten.nn.from_pytorch(model_plaintext, dummy_input)
    model.encrypt()

    loss = crypten.nn.CrossEntropyLoss()
    # loss = crypten.nn.MSELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    num_epoch = 500
    learning_rate = 0.05

    rank = crypten.communicator.get().get_rank()
    print("My rank is", rank)  # debug point

    label_eye = torch.eye(43)
    for epoch in range(num_epoch):

        model.cpu()
        model.train()

        for i, data in enumerate(train_loader):

            model.zero_grad()
            site0_feature_enc = crypten.cryptensor(data[0], src=0)
            # site1_feature_enc = crypten.cryptensor(data[0], src=1)
            # site2_feature_enc = crypten.cryptensor(data[0], src=2)
            # site3_feature_enc = crypten.cryptensor(data[0], src=3)

            # x_combined_enc = crypten.cat(
            #     [site0_feature_enc, site1_feature_enc, site2_feature_enc, site3_feature_enc], dim=0)
            x_combined_enc = crypten.cat([site0_feature_enc], dim=0)

            x_combined_enc.requires_grad = True

            site0_label_plain = data[1]
            y_one_hot = label_eye[site0_label_plain]
            y_train = crypten.cryptensor(y_one_hot, requires_grad=True)

            # model.cuda()
            # x_combined_enc.cuda()
            # y_train.cuda()
            # loss.cuda()

            # pdb.set_trace()
            train_pred = model(x_combined_enc)

            batch_loss = loss(train_pred, y_train)

            batch_loss.backward()
            model.update_parameters(learning_rate)

        model.eval()
        model.cuda()

        # Compute training accuracy every epoch
        correct_count = 0
        loss_sum = 0
        for i, data in enumerate(train_loader):
            site0_feature_enc = crypten.cryptensor(data[0], src=0)
            x_combined_enc = crypten.cat([site0_feature_enc], dim=0)
            x_combined_enc.requires_grad = False

            site0_label_plain = data[1]
            y_one_hot = label_eye[site0_label_plain]
            y_train = crypten.cryptensor(y_one_hot, requires_grad=True)

            train_pred = model(x_combined_enc.cuda())
            train_pred.cpu()

            batch_loss = loss(train_pred, y_train)

            pred = train_pred.get_plain_text().argmax(1)
            correct = pred.eq(site0_label_plain)
            correct_count += correct.sum(0, keepdim=True).float()

            loss_plaintext = batch_loss.get_plain_text().item()
            loss_sum += loss_plaintext

        accuracy = correct_count / train_set.__len__()
        print(
            f"Rank {rank} Epoch {epoch} Training: "
            f"Loss {loss_sum:.4f} Accuracy {accuracy.item():.2f}"
        )

        # Compute validation accuracy every epoch
        correct_count = 0
        loss_sum = 0
        for i, data in enumerate(valid_loader):
            site0_feature_enc = crypten.cryptensor(data[0], src=0)
            x_combined_enc = crypten.cat([site0_feature_enc], dim=0)
            x_combined_enc.requires_grad = False

            site0_label_plain = data[1]
            y_one_hot = label_eye[site0_label_plain]
            y = crypten.cryptensor(y_one_hot, requires_grad=True)

            val_pred = model(x_combined_enc.cuda())
            val_pred.cpu()

            batch_loss = loss(val_pred, y)

            pred = val_pred.get_plain_text().argmax(1)
            correct = pred.eq(site0_label_plain)
            correct_count += correct.sum(0, keepdim=True).float()

            loss_plaintext = batch_loss.get_plain_text().item()
            loss_sum += loss_plaintext

        accuracy = correct_count / valid_set.__len__()
        print(
            f"Rank {rank} Epoch {epoch} Validation: "
            f"Loss {loss_sum:.4f} Accuracy {accuracy.item():.2f}"
        )


if __name__ == "__main__":
    run_mpc_cnn()
