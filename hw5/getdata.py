from torchvision.datasets import CIFAR100

data = CIFAR100(root="~/Dataset/CIFAR100", train=True)
p, target = data.__getitem__(5)
p.show()
print(target)

