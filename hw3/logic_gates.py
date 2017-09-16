import numpy as np
from neural_network import NeuralNetwork



class Dataset():
    def __init__(self, gate):
        if gate == "AND":
            self.data = np.array([[1,1,1,1],[1,1,0,0],[1,0,1,0],[1,0,0,0]])
        
        if gate == "OR":
            self.data = np.array([[1,1,1,1],[1,1,0,1],[1,0,1,1],[1,0,0,0]])
        
        if gate == "NOT":
            self.data = np.array([[1,1,0],[1,0,1]])

        if gate == "XOR":
            self.data = np.array([[1,0,0,0],[1,0,1,1],[1,1,0,1],[1,1,1,0]])

        self.features = self.data[:,0:-1]
        self.label = self.data[:,-1].reshape(-1, 1)

    def shuffle(self):
        np.random.shuffle(self.data)



class And(NeuralNetwork):
    def __init__(self, shape=[3,3,1]):
        NeuralNetwork.__init__(self, shape=shape)

    def forward(self, Tensor):
        NeuralNetwork.forward(Tensor)

    def backward(self, DTensor):
        NeuralNetwork.backward(Tensor)

    def train(self, eta):
        data = Dataset("AND")


a = And()
print(a.Theta)
         
