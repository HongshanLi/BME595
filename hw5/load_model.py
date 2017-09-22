import torch
from img2obj import LeNet
from torchvision.datasets import CIFAR100
from torchvision import transforms
from torch.autograd import Variable

data = CIFAR100(root="~/Dataset/CIFAR100",
        transform=transforms.Compose([transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]))

x, y = data.__getitem__(5)
x = x.view(1, 3, 32, 32)
x = Variable(x)

b = LeNet()
b.load_state_dict(torch.load("latest_parameters.pt"))
b.evaluate()

