from torchvision.datasets import CIFAR100
from torchvision import transforms
from torch.autograd import Variable

data = CIFAR100(root="~/Dataset/CIFAR100", download=False, 
        train=True, transform=transforms.ToTensor())


p, target = data.__getitem__(5)

print(p.shape)
print(type(p))

