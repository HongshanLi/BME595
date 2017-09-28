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
import numpy as np
import pickle
import cv2
from PIL import Image
import matplotlib.pyplot as plt

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
parser.add_argument("--num_epoch", type=int, default=100, metavar="N",
    help="number of epochs to train")

args = parser.parse_args()

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        with open("./meta", "rb") as f:
            self.fine_label_names = pickle.load(f)["fine_label_names"]    

        self.conv1 = nn.Conv2d(3,6,5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5, stride=2, padding=2)
        self.fc1 = nn.Linear(16*8*8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 100)
        self.optim = optim.SGD(self.parameters(), lr=args.lr, momentum=args.momentum)
        self.training_loss = []
        self.accuracy = []


    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.view(-1, 3, 32, 32).float()
        x = Variable(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 8*8*16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x)

    def view(self, img):
        # Argument:
        #   img: 3x32x32 tensor in uint8
        # Return:
        #   image and the prediction of its class
        pic = Image.fromarray(img)
        pic.show()
        img = img.astype(np.float32)
        img = img / 255
        pred = self.forward(img)
        pred = pred.view(-1)
        values, indices = pred.max(0)
        indices = np.asscalar(indices.data.numpy())
        print(self.fine_label_names[indices])

    def cam(self):
        cap = cv2.VideoCapture(0)
        while(True):
            ret, frame = cap.read()
            cv2.imshow('frame',frame)
            
            ch = cv2.waitKey(1)
            if ch == ord("c"):
                pic = frame
                pic = cv2.resize(pic, (32, 32))
                pic = np.array(pic)
                pic = pic / 255
                pred = self.forward(pic).view(-1)
                values, indices = pred.max(0)
                indices = np.asscalar(indices.data.numpy())
                print(self.fine_label_names[indices])

            if ch == ord("q"):
                break
        cap.release()
        cap.destroyAllWindow()
        

    def train(self):
        print("Starting a new epoch with learning rate: %f" % args.lr)
    
        train_data = CIFAR100(root="~/Dataset/CIFAR100", train=True,
            transform=transforms.Compose([transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]))
        train_data_loader = DataLoader(train_data, batch_size = args.batch_size,
            shuffle=True)

        current_loss = 0
        for batch_idx, (data, target) in enumerate(train_data_loader):
            target = Variable(target)
            self.optim.zero_grad()
            pred = self.forward(data)
            loss = F.nll_loss(pred, target)
            loss.backward()
            self.optim.step()
            current_loss += loss

            if batch_idx % args.log_interval==0:
                print("Step: %d, negative log loss %f" % (batch_idx, loss.data[0]))
        
        # compute the average training loss
        current_loss = current_loss.data.numpy()
        current_loss = np.asscalar(current_loss)
        self.training_loss.append(current_loss / len(train_data_loader))        
        

    

    def evaluate(self):              
        # Accuracy of the model after the current epoch
        test_data = CIFAR100(root="~/Dataset/CIFAR100", train=False,
            transform=transforms.Compose([transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]))
        test_data_loader = DataLoader(test_data, batch_size=len(test_data),
            shuffle=False)
        for data, target in test_data_loader:
            target = Variable(target)
            pred = self.forward(data)
            pred = pred.data.max(1, keepdim=True)[1]
            correct = pred.eq(target.data.view_as(pred)).cpu().sum()

        self.accuracy.append(float(correct) / float(len(test_data)))
        print("The accuracy is:%f" % self.accuracy[-1])

    def summarize(self):
        num_epoch = len(self.accuracy)
        epoch = range(1, num_epoch+1)
        # save training loss
        training_loss = np.array(self.training_loss)
        np.save("TrainingLoss.npy", training_loss)
        test_accuracy = np.array(self.accuracy)
        np.save("TestAccuracy.npy", test_accuracy)

        plt.scatter(epoch, self.training_loss, s=10, c="b", 
            marker="s", label="Average Training Error")
        plt.legend(loc="upper left")
        plt.savefig("AverageTrainingError.png")
        plt.gcf().clear()

        plt.scatter(epoch, self.accuracy, s=10, c="r",
            marker="s", label="Accuracy on Test Set")
        plt.legend(loc="upper left")
        plt.savefig("AccuracyOnTestSet.png")
        plt.gcf().clear()

        


            

            

if __name__ == "__main__":
    a = LeNet()
 
    for epoch in range(1, args.epoch+1):
        if epoch % 10 ==1 and epoch > 10:
            args.lr = args.lr / 10       
        a.train()
        torch.save(a.state_dict(), "latest_parameters.pt")
        a.evaluate()
    
    a.summarize()
    

