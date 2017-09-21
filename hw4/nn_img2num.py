from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class img2num(nn.Module):
    def __init__(self):
        super(img2num, self).__init__()
        self.fc1 = nn.Linear(784, 50)
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 10)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.log_softmax(self.fc3(x))
        return 

    def train(self):
        batch_size = 100
        optimizer = optim.SGD(self.parameters(), lr=0.1)

        dataset = MNIST(root="/homes/li108/Dataset/Mnist", train=True,
                        transform=transforms.ToTensor())
        
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                
        for batch_index, (data, target) in enumerate(train_loader):
            data = data.view(batch_size, -1)
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = self.forward(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()




class img2num(NeuralNetwork):
    def __init__(self):
        NeuralNetwork.__init__(self, shape=[784, 50, 20,10])
        self.average_training_error=[] # keeps track of average training error per epoch


    def train(self):
        learning_rate = 0.1
        dataset = MNIST(root="/homes/li108/Dataset/Mnist", train=True,
                         transform=transforms.ToTensor())
        num_epoch = 1
        batch_size = 100
        num_batch_per_epoch = len(dataset) / batch_size
    
        # after each epoch shuffle the data
    
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
        # convert features into flat numpy array but keep batch
        def flatten_each_input(batch):
            A = batch.numpy()
            A = A.flatten()
            A = A.reshape(batch_size, -1)
            return A

        def one_hot_encode_label(batch_target):
            batch_label = np.zeros([batch_size, 10])
            batch_label[np.arange(batch_size), batch_target.numpy()] = 1
            return batch_label

        for epoch in range(num_epoch):
            epoch_error = 0
            for batch_index, (features, target) in enumerate(train_loader):
                features = flatten_each_input(features)
                one_hot_labels = one_hot_encode_label(target)
                self.forward(features)
                self.backward(one_hot_labels)
                self.updateParams(learning_rate)
                epoch_error += self.current_error

            self.average_training_error.append(epoch_error / num_batch_per_epoch)
        
        print(self.average_training_error)
        

n = img2num()
n.train()

    
        
