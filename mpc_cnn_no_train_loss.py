#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from crypten import mpc
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
from torch.utils.data.dataset import Dataset


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

        # Input: [3, 48, 48]

        # Conv Layers
        out = self.cnn1(x)  # [8, 48, 48]
        out = F.relu(out)
        out = F.max_pool2d(out, 4)  # [8, 12, 12]

        # Flattern
        out = out.view(-1, 8 * 12 * 12)  # [8*12*12]

        # FC Layers
        out = self.fc1(out)  # [43]

        # Ouput: [43]
        return out


@mpc.run_multiprocess(world_size=4)
def run_mpc_cnn(
    epochs=50, examples=50, features=100, lr=0.05, skip_plaintext=False
):

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


class CustomDataset(Dataset):

    def __init__(self, x_enc, y_enc):
        self.x_enc = x_enc
        self.y_enc = y_enc

    def __len__(self):
        return self.x_enc.shape[0]

    def __getitem__(self, idx):
        return self.x_enc[idx], self.y_enc[idx]


def do_train(training_datastats, validation_datastats, resize_hw=48):
    rank = crypten.communicator.get().get_rank()
    print("My rank is", rank)  # debug point

    data_size = [252, 284, 279, 231]  # Known by inspecting data
    batch_size = 32

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

    x_site = torch.empty([0, 3, 48, 48])
    y_site = torch.empty([0])
    for data in train_loader:
        x_site = torch.cat((x_site, torch.tensor(data[0])), dim=0)
        y_site = torch.cat((y_site, torch.tensor(data[1])), dim=0)

        # crypten.save(features_alice, filenames["features_alice"], src=ALICE)
    print(x_site.shape)
    print(y_site.shape)

    if rank == 0:
        x_site0 = x_site
        y_site0 = y_site
    else:
        x_site0 = torch.empty([data_size[0], 3, 48, 48])
        y_site0 = torch.empty([data_size[0]])

    if rank == 1:
        x_site1 = x_site
        y_site1 = y_site
    else:
        x_site1 = torch.empty([data_size[1], 3, 48, 48])
        y_site1 = torch.empty([data_size[1]])

    if rank == 2:
        x_site2 = x_site
        y_site2 = y_site
    else:
        x_site2 = torch.empty([data_size[2], 3, 48, 48])
        y_site2 = torch.empty([data_size[2]])

    if rank == 3:
        x_site3 = x_site
        y_site3 = y_site
    else:
        x_site3 = torch.empty([data_size[3], 3, 48, 48])
        y_site3 = torch.empty([data_size[3]])

    x_site0_enc = crypten.cryptensor(x_site0, src=0)
    x_site1_enc = crypten.cryptensor(x_site1, src=1)
    x_site2_enc = crypten.cryptensor(x_site2, src=2)
    x_site3_enc = crypten.cryptensor(x_site3, src=3)

    y_site0_enc = crypten.cryptensor(y_site0, src=0)
    y_site1_enc = crypten.cryptensor(y_site1, src=1)
    y_site2_enc = crypten.cryptensor(y_site2, src=2)
    y_site3_enc = crypten.cryptensor(y_site3, src=3)

    x_sites_enc = crypten.cat(
        [x_site0_enc, x_site1_enc, x_site2_enc, x_site3_enc], dim=0)
    y_sites_enc = crypten.cat(
        [y_site0_enc, y_site1_enc, y_site2_enc, y_site3_enc], dim=0)

    train_dataset = CustomDataset(x_sites_enc, y_sites_enc)
    print("len train_dataset", len(train_dataset))

    dummy_input = torch.randn(1, 3, resize_hw, resize_hw)

    model_plaintext = Classifier()
    model = crypten.nn.from_pytorch(model_plaintext, dummy_input)
    model.encrypt()

    loss = crypten.nn.CrossEntropyLoss()
    # loss = crypten.nn.MSELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    num_epoch = 200
    learning_rate = 0.05

    label_eye = torch.eye(43)

    num_samples = x_sites_enc.shape[0]

    y_sites_plain = y_sites_enc.get_plain_text().long()

    for epoch in range(num_epoch):

        model.cpu()
        model.train()

        for j in range(0, num_samples, batch_size):

            start, end = j, min(j + batch_size, num_samples)

            model.zero_grad()
            x_sites_enc.requires_grad = True

            label_one_hot = label_eye[y_sites_plain[start:end]]
            y_enc_batch = crypten.cryptensor(label_one_hot)
            y_enc_batch.requires_grad = True

            # TODO: y_sites_enc.requires_grad, not batchly part

            # model.cuda()
            # x_combined_enc.cuda()
            # y_train.cuda()
            # loss.cuda()

            # pdb.set_trace()

            # y_pred = model(x_combined_enc)
            y_pred = model(x_sites_enc[start:end])

            batch_loss = loss(y_pred, y_enc_batch)

            batch_loss.backward()
            model.update_parameters(learning_rate)

            print("Epoch", epoch, "Batch", int(j/batch_size))

        model.eval()

        # # Compute training accuracy every epoch
        # correct_count = 0
        # loss_sum = 0
        # for i, data in enumerate(train_loader):
        #     site0_feature_enc = crypten.cryptensor(data[0], src=0)
        #     x_combined_enc = crypten.cat([site0_feature_enc], dim=0)
        #     x_combined_enc.requires_grad = False

        #     site0_label_plain = data[1]
        #     y_one_hot = label_eye[site0_label_plain]
        #     y_train = crypten.cryptensor(y_one_hot, requires_grad=True)

        #     train_pred = model(x_combined_enc)

        #     batch_loss = loss(train_pred, y_train)

        #     pred = train_pred.get_plain_text().argmax(1)
        #     correct = pred.eq(site0_label_plain)
        #     correct_count += correct.sum(0, keepdim=True).float()

        #     loss_plaintext = batch_loss.get_plain_text().item()
        #     loss_sum += loss_plaintext

        # accuracy = correct_count / train_set.__len__()
        # print(
        #     f"Rank {rank} Epoch {epoch} Training: "
        #     f"Loss {loss_sum:.4f} Accuracy {accuracy.item():.2f}"
        # )

        # Compute validation accuracy every epoch

        model.cuda()

        correct_count = 0
        loss_sum = 0
        for i, data in enumerate(valid_loader):
            site0_feature_enc = crypten.cryptensor(data[0], src=0)
            x_combined_enc = crypten.cat([site0_feature_enc], dim=0)
            x_combined_enc.requires_grad = False

            site0_label_plain = data[1]
            y_one_hot = label_eye[site0_label_plain]
            y = crypten.cryptensor(y_one_hot, requires_grad=True)

            val_pred = model(x_combined_enc.cuda()).cpu()

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
