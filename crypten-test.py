import pdb
import crypten

import tempfile

import crypten
import crypten.communicator as comm
import torch
import torch.nn as nn
import torch.nn.functional as F
from examples.util import NoopContextManager
from torchvision import datasets, transforms


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=0)
        self.fc1 = nn.Linear(16 * 12 * 12, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = F.max_pool2d(out, 2)
        out = out.view(-1, 16 * 12 * 12)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out


# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.fc1 = nn.Linear(3, 100)
#         self.fc2 = nn.Linear(100, 2)
#
#     def forward(self, x):
#         out = self.fc1(x)
#         out = F.relu(out)
#         out = self.fc2(out)
#         return out

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


crypten.init()

# a = torch.tensor([[1, 2, 3], [4, 5, 6]])
# print(a)
# b = crypten.cryptensor(a, src=0)
# print(b)
# c = crypten.cat([b, b], dim=0)
# print(c)
#
#
# dummy_input = torch.randn(1, 1, 28, 28)
# print(dummy_input.shape)
#
# model_plaintext = CNN()
# print(model_plaintext)
# model = crypten.nn.from_pytorch(model_plaintext, dummy_input)
# model.train()
# model.encrypt()
#
# real_input = crypten.cryptensor(torch.randn(5, 1, 28, 28), src=0)
#
# output = model(real_input)
# print(output)
# print(output.get_plain_text())


dummy_input = torch.randn(1, 3, 48, 48)

print(dummy_input.shape)
# pdb.set_trace()

model_plaintext = Classifier()
model = crypten.nn.from_pytorch(model_plaintext, dummy_input)
model.encrypt()

real_input = crypten.cryptensor(torch.randn(32, 3, 48, 48), src=0)

real_input.cuda()
model.cuda()

pdb.set_trace()

print(real_input.shape)
output = model(real_input)
print(output.shape)
