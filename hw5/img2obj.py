import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision.datasets import CIFAR100
from torchvision import transforms
from torch.utils.data import DataLoader
from time import time
import argparse

parser = argparse.ArgumentParser(description='Lenet on CIFAR100')
parser.add_argument("--batch_size", type=int, default=128, metavar="N",
    help="mini-batch size")
parser.add_argument("--epoch", type=int, default=10, metavar="N",
    help="number of epochs to train")
parser.add_argument("--lr", type=float, default=0.1, metavar="LR",
    help="learning rate")
parser.add_argument("--momentum", type=float, default=0.5, metavar="M",
    help="SGD momentum")
parser.add_argument("--seed", type=int, default=5, metavar="S",
    help="random seed to for initialization")
parser.add_argument("--log_interval", type=int, default=100, metavar="N",
    help="number of steps to print out one log")
args = parser.parse_args()

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3,6,5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=2)
        self.fc1 = nn.Linear(16*8*8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 100)
        self.optim = optim.SGD(self.parameters(), lr=args.lr, 
            momentum=args.momentum)
        self.training_loss = 0
        self.test_loss = 0
        self.accuracy = 0

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 8*8*16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x)

    def train(self):
        print("Starting a new epoch with learning rate: %f" % args.lr)
    
        train_data = CIFAR100(root="~/Dataset/CIFAR100", train=True,
            transform=transforms.Compose([transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]))
        train_data_loader = DataLoader(train_data, batch_size = args.batch_size,
            shuffle=True)

        
        for batch_idx, (data, target) in enumerate(train_data_loader):
            data, target = Variable(data), Variable(target)
            self.optim.zero_grad()
            pred = self.forward(data)
            loss = F.nll_loss(pred, target)
            loss.backward()
            self.optim.step()
            self.training_loss += loss

            if batch_idx % args.log_interval==0:
                print("Step: %d, negative log loss %f" % (batch_idx, loss.data[0]))
        
        # compute the average training loss
        self.training_loss = self.training_loss / len(train_data_loader)        
        
        # log parameters of current epoch
        torch.save(self.state_dict(), "latest_parameters.pt")

    def evaluate(self):              
        # Accuracy of the model after the current epoch
        test_data = CIFAR100(root="~/Dataset/CIFAR100", train=False,
            transform=transforms.Compose([transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]))
        test_data_loader = DataLoader(test_data, batch_size=len(test_data),
            shuffle=False)
        for data, target in test_data_loader:
            data, target = Variable(data, volatile=True), Variable(target)
            pred = self.forward(data)
            self.test_loss = F.nll_loss(pred, target, size_average=False).data[0]
            pred = pred.data.max(1, keepdim=True)[1]
            correct = pred.eq(target.data.view_as(pred)).cpu().sum()

        self.accuracy = float(correct) / float(len(test_data))
        print("The accuracy is:%f" % self.accuracy)

            

            
if __name__=="__main__":
    initial_learning_rate = args.lr
    a = LeNet()
    f = open("training_log", "w+")
    for epoch in range(2):
        args.lr = initial_learning_rate / (epoch + 1)
        a.train()
        a.evaluate()
        f.write("Epoch" + str(epoch) + ',' "Average training loss:" + str(a.training_loss) + ",")
        f.write("Average test loss:" + str(self.test_loss) + "," + "Accuracy:" + str(self.accuracy))

    f.close()
