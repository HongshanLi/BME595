from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from time import time
import matplotlib.pyplot as plt


class img2num(nn.Module):
    def __init__(self):
        super(img2num, self).__init__()
        self.fc1 = nn.Linear(784, 50)
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 10)
        self.epoch = []
        self.epoch_accuracy = []
    
    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.log_softmax(self.fc3(x))
        return x 

    def train(self):
        start_time = time()
        num_epoch = 5
        batch_size = 100
        optimizer = optim.SGD(self.parameters(), lr=0.1)
        log_interval = 100
        epoch = 1
        train_dataset = MNIST(root="/homes/li108/Dataset/Mnist", train=True,
                        transform=transforms.ToTensor())
        
        test_dataset = MNIST(root="/homes/li108/Dataset/Mnist", train=False,
                        transform=transforms.ToTensor())

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(num_epoch):
            self.epoch.append(epoch+1)
            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.view(batch_size, -1)
                data, target = Variable(data), Variable(target)
                optimizer.zero_grad()
                output = self.forward(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()


        # test accuracy on test set
            print("Now testing accuracy")
            test_loss = 0
            correct = 0
            for data, target in test_loader:
                data = data.view(batch_size, -1)
                data, target = Variable(data, volatile=True), Variable(target)
                output = self.forward(data)
                test_loss+=F.nll_loss(output, target, size_average=False).data[0]
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            test_loss /= len(test_loader.dataset)
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{}({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset), 
                100. * correct/ len(test_loader.dataset)))
            
            accuracy = float(correct) / len(test_loader.dataset) 
            self.epoch_accuracy.append(accuracy)

        end_time = time()
        print("total run time: %f" % (end_time - start_time))




n = img2num()
n.train()
plt.scatter(x=n.epoch, y=n.epoch_accuracy)
plt.savefig("nn_img2num_performance.png")

    
        
