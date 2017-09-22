import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision.datasets import CIFAR100
from torchvision import transforms
from time import time


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3,6,5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=2)
        self.fc1 = nn.Linear(16*8*8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 100)

    def forward(self, x):
        print(x.shape)
        x = F.relu(self.conv1(x))
        print(x.shape)
        x = F.max_pool2d(x, 2)
        print(x.shape)
        x = F.relu(self.conv2(x))
        print(x.shape)
        x = F.max_pool2d(x, 2)
        print(x.shape)
        x = x.view(-1, 8*8*16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

data = CIFAR100(root="~/Dataset/CIFAR100",transform=transforms.ToTensor())
x, y = data.__getitem__([0:100])
x = x.view(100, 3, 32, 32)
x = Variable(x)
start_time = time()

Net = LeNet()
Net.forward(x)
end_time = time()
run_time = end_time - start_time
print(run_time)       
