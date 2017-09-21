from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
from neural_network import NeuralNetwork
import numpy as np
from time import time
import matplotlib.pyplot as plt

class img2num(NeuralNetwork):
    def __init__(self):
        NeuralNetwork.__init__(self, shape=[784, 50, 10])
        self.epoch = []
        self.average_training_error=[] # keeps track of average training error per epoch
        self.epoch_accuracy = [] 

    def train(self):
        learning_rate = 0.1
        train_dataset = MNIST(root="/homes/li108/Dataset/Mnist", train=True,
                         transform=transforms.ToTensor())
        test_dataset = MNIST(root="/homes/li108/Dataset/Mnist", train=True,
                        transform=transforms.ToTensor())

        num_epoch = 5
        batch_size = 100
        num_batch_per_epoch = len(train_dataset) / batch_size
    
        #after each epoch shuffle the 
    
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False)
        
        # convert features into flat numpy array but keep batch
        def flatten_each_input(batch):
            A = batch.numpy()
            batch_size = A.shape[0]
            A = A.flatten()
            A = A.reshape(batch_size, -1)
            return A

        def one_hot_encode_label(batch_target):
            batch_label = np.zeros([batch_size, 10])
            batch_label[np.arange(batch_size), batch_target.numpy()] = 1
            return batch_label
        
        start_time = time()
        for epoch in range(num_epoch):
            self.epoch.append(epoch+1)
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
        

            print("Now testing")
            correct = 0
            for batch_idx, (features, target) in enumerate(test_loader):
                features = flatten_each_input(features)
                pred = self.forward(features)
                pred = np.argmax(pred, axis=1)
                correct += np.equal(pred, target.numpy()).sum()
             
            accuracy = float(correct) / len(test_loader.dataset)
            self.epoch_accuracy.append(accuracy)   

        end_time = time() 
        print("run time %f:" % (end_time - start_time)) 

n = img2num()
n.__init__()
n.train()
print(n.epoch_accuracy)
plt.scatter(x=n.epoch, y=n.epoch_accuracy)
plt.savefig("my_img2num_performance.png")
plt.show()
    
        
