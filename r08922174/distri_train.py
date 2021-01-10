import os
import cv2
import numpy as np
import pdb
import math
import argparse
import random

import torch
import torchvision
import torch.nn.functional as F
from torch import nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, required=True, help="GPU id that this program used.")
parser.add_argument('--train_path', type=str, default='./splitdata_small/train/total/')
parser.add_argument('--validation_path', type=str, default='./splitdata_small/validation/')
parser.add_argument('--test_path', type=str, default='./GTSRB_Challenge/test/')
parser.add_argument('--model_path', type=str, default='./')
parser.add_argument('--model_load', type=str, default='pretrained_model.pt')
parser.add_argument('--model_save', type=str, default='./savedModels/nice_distributed.pth')
parser.add_argument('--ep', type=int, default=100)
parser.add_argument('--size', type=int, default=48)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--train_val_split_ratio', type=float, default=0.9)
parser.add_argument('--bs', type=int, default=32)
parser.add_argument('--loadOrNot', type=int, default=0, help="1: load saved model, others: no loading.")
parser.add_argument('--show', type=int, default=1)
parser.add_argument('--site', type=int, default=0)
parser.add_argument("--resume", action="store_true", help="Resume training from saved checkpoint.", default=False)
args = parser.parse_args()


def init_random_seed(seed=0):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.

def statistic(DataPath):
    
    transform = transforms.Compose([
        transforms.Resize((args.size, args.size)), 
        transforms.ToTensor(), 
    ])
    dataset = ImageFolder(DataPath, transform = transform)
    total = 43

    count = torch.IntTensor([0] * total)
    mean = torch.stack([image.mean((1, 2)) for image, _ in dataset]).mean(0)
    std = torch.stack([image for image, _ in dataset]).std((0, 2, 3))
    for label in range(total):
        count[label] = len(os.listdir(DataPath + '%05d/'%label))

    del transform, dataset

    return tuple(mean), tuple(std), count

def evaluate(model, device, test_loader):

    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total

    return accuracy

#  class Classifier(nn.Module):
#      def __init__(self):
#          super(Classifier, self).__init__()
#  
#          self.cnn = nn.Sequential(
#              nn.Conv2d(3, 64, 3, 1, 1),  # [64, 48, 48]
#              nn.BatchNorm2d(64),
#              nn.ReLU(),
#              nn.MaxPool2d(2, 2, 0),      # [64, 24, 24]
#  
#              nn.Conv2d(64, 128, 3, 1, 1),  # [128, 24, 24]
#              nn.BatchNorm2d(128),
#              nn.ReLU(),
#              nn.MaxPool2d(2, 2, 0),      # [128, 12, 12]
#  
#              nn.Conv2d(128, 256, 3, 1, 1),  # [256, 12, 12]
#              nn.BatchNorm2d(256),
#              nn.ReLU(),
#              nn.Conv2d(256, 256, 3, 1, 1),  # [256, 12, 12]
#              nn.BatchNorm2d(256),
#              nn.ReLU(),
#              nn.MaxPool2d(2, 2, 0),      # [256, 6, 6]
#          )
#          self.fc = nn.Sequential(
#              # nn.Linear(512*4*4, 1024),
#              nn.Linear(256*6*6, 1024),
#              nn.ReLU(),
#              nn.Linear(1024, 512),
#              nn.ReLU(),
#              nn.Linear(512, 43)
#          )
#  
#      def forward(self, x):
#          out = self.cnn(x)
#          out = out.view(out.size()[0], -1)  # Flattern
#          return self.fc(out)
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


def do_train():

    mean, std, count = statistic(args.train_path)
    #  print("Mean: {}, Standard deviation: {}, count: {}".format(mean, std, count))

    custom_transform = transforms.Compose(
        [transforms.Resize([args.size, args.size]),
         transforms.ToTensor(), 
         transforms.Normalize(mean, std)]
    )
    #  total = sum(count)
    #  weight = []
    #  for number in count:
    #      weight += [torch.true_divide(total, number)] * number
    #  #  print("Check weight length: ", len(weight))
    #  #  print("Check total length: ", int(total))
    #  sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=torch.Tensor(weight), num_samples=int(total))
    #  del count, weight

    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.distributed.init_process_group(backend="nccl")
    if args.loadOrNot == 1:
        model = torch.load(args.model_path + args.model_load)
    else:
        model = Classifier().cuda()

    device = torch.device("cuda:{}".format(args.local_rank))
    model = model.to(device)
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    # We only save the model who uses device "cuda:0"
    # To resume, the device for the saved model would also be "cuda:0"
    if args.resume == True:
        map_location = {"cuda:0": "cuda:{}".format(args.local_rank)}
        ddp_model.load_state_dict(torch.load(model_filepath, map_location=map_location))

    train_set = ImageFolder(root=args.train_path, transform=custom_transform)
    val_set = ImageFolder(root=args.validation_path, transform=custom_transform)
    #  test_set = ImageFolder(root=args.test_path, transform=custom_transform)

    train_sampler = DistributedSampler(dataset=train_set)

    train_loader = DataLoader(train_set, batch_size=args.bs, \
            pin_memory=True, drop_last=True, sampler=train_sampler)
    val_loader = DataLoader(val_set, batch_size=args.bs)
    #  test_loader = DataLoader(test_set, batch_size=args.bs)

    LossFunction = nn.CrossEntropyLoss()
    #  optimizer = torch.optim.Adam(ddp_model.parameters(), lr=args.lr)
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.05)
    #  optimizer = torch.optim.SGD(
    #      ddp_model.parameters(), lr=args.lr, momentum=0.9)

    BestAccuracy = 0.0

    if args.model_save:
            torch.save(ddp_model, args.model_path + args.model_save)
    
    for epoch in range(args.ep):
        ddp_model.train()
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            train_pred = ddp_model(data[0].cuda())
            batch_loss = LossFunction(train_pred, data[1].cuda())
            batch_loss.backward()
            optimizer.step()

        train_loss = 0.0
        train_acc = 0.0
        ddp_model.eval()
        for i, data in enumerate(train_loader):
            train_pred = ddp_model(data[0].cuda())
            batch_loss = LossFunction(train_pred, data[1].cuda())
            train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(),
                                          axis=1) == data[1].numpy())
            train_loss += batch_loss.item()

        val_acc = 0.0
        ddp_model.eval()
        for i, data in enumerate(val_loader):
            val_pred = ddp_model(data[0].cuda())
            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(),
                                        axis=1) == data[1].numpy())
        accuracy = val_acc / len(val_set)
        print('Local rank: %d, Epoch: %d, Train Acc: %3.6f Loss: %3.6f Val Acc: %3.6f' %
              (args.local_rank, epoch, train_acc / len(train_set), train_loss / len(train_set), accuracy))
        if args.model_save and accuracy > BestAccuracy:
            BestAccuracy = accuracy
            torch.save(ddp_model, args.model_path + args.model_save)

def main():

    init_random_seed(seed=0)
    do_train()

if __name__ == "__main__":
    main()
