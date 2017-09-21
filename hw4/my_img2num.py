from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
from neural_network import NeuralNetwork
import numpy as np



class img2num(NeuralNetwork):
    def __init__(self):
        NeuralNetwork.__init__(self, shape=[784, 50, 20, 10])
        self.average_training_error=[] # keeps track of average training error per epoch


    def train(self):
        learning_rate = 0.1
        dataset = MNIST(root="/homes/li108/Dataset/Mnist", train=True,
                         transform=transforms.ToTensor())
        num_epoch = 1
        batch_size = 100
        num_batch_per_epoch = len(dataset) / batch_size
    
        #after each epoch shuffle the 
    
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

    
        
