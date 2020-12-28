import os
import cv2
import numpy as np
import pdb
import math

import torch
from torch import nn
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


class TestingDataSet(DataSet):
    def __init__(self, dataset_path):
        DataSet.__init__(self)
        self.fetch_data_stats(dataset_path)

    def fetch_data_stats(self, current_dir):

        image_files = os.listdir(current_dir)

        total_metrics = np.zeros((len(image_files), 2), dtype=int)

        for i, image_file in enumerate(image_files, start=0):
            image_path = os.path.join(current_dir, image_file)
            img = cv2.imread(image_path)

            total_metrics[i, 0] = img.shape[0]
            total_metrics[i, 1] = img.shape[1]

        self._path = current_dir
        self._num_of_data = total_metrics.shape[0]
        self._mean_h, self._mean_w = np.mean(total_metrics, axis=0)
        self._stddev_h, self._stddev_w = np.std(total_metrics, axis=0)
        self._max_h, self._max_w = np.amax(total_metrics, axis=0)

# Model


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


def do_train(training_dataset):

    from torchvision.datasets import ImageFolder

    custom_transform = transforms.Compose(
        [transforms.Resize([48, 48]),
         transforms.ToTensor()]
    )
    dataset = ImageFolder(training_dataset.path, custom_transform)

    num_of_train_set = math.floor(0.9 * len(dataset))
    num_of_val_set = len(dataset) - num_of_train_set

    train_set, val_set = torch.utils.data.random_split(
        dataset, [num_of_train_set, num_of_val_set])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=True)

    model = Classifier().cuda()
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.001, momentum=0.9)
    num_epoch = 100

    for epoch in range(num_epoch):
        model.train()
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            train_pred = model(data[0].cuda())
            batch_loss = loss(train_pred, data[1].cuda())
            batch_loss.backward()
            optimizer.step()

        train_loss = 0.0
        train_acc = 0.0
        model.eval()
        for i, data in enumerate(train_loader):
            train_pred = model(data[0].cuda())
            batch_loss = loss(train_pred, data[1].cuda())
            train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(),
                                          axis=1) == data[1].numpy())
            train_loss += batch_loss.item()

        val_acc = 0.0
        model.eval()
        for i, data in enumerate(val_loader):
            val_pred = model(data[0].cuda())
            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(),
                                        axis=1) == data[1].numpy())

        print('Train Acc: %3.6f Loss: %3.6f Val Acc: %3.6f' %
              (train_acc/len(train_set), train_loss/len(train_set), val_acc/len(val_set)))


def init_random_seed(seed=0):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.


def main():

    init_random_seed(seed=0)

    # training_dataset_path = "/home/r06922149/download/acc-german/GTSRB_Challenge/train"
    # testing_dataset_path = "/home/r06922149/download/acc-german/GTSRB_Challenge/test"

    training_dataset_path = "./data/train"
    testing_dataset_path = "./data/test"

    testing_dataset = TestingDataSet(testing_dataset_path)
    training_dataset = TrainingDataSet(training_dataset_path)

    testing_dataset.print_stats()
    training_dataset.print_stats()

    do_train(training_dataset)


if __name__ == "__main__":
    main()
