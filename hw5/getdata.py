from torchvision.datasets import CIFAR100
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import torch
from img2obj import LeNet

data = CIFAR100(root="~/Dataset/CIFAR100", download=False,
        train=True)


p, target = data.__getitem__(5)
n = LeNet()

print(n.view(p))




