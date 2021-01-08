import torch.nn
import torch.nn.functional

class Classifier(torch.nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.cnn = torch.nn.Conv2d(3, 8, kernel_size = 3, stride = 1, padding = 1)
        self.fc = torch.nn.Linear(8 * 12 * 12, 43)

    def forward(self, x):
        return self.fc(torch.nn.functional.max_pool2d(torch.nn.functional.relu(self.cnn(x)), 4).reshape(-1, 8 * 12 * 12))