import numpy as np
import math

class NeuralNetwork():
    def __init__(self, shape=list()):
        self.shape = shape
        self.weights = []
        for i in range(len(shape)-1):
            self.weights.append(np.random.rand(shape[i+1], shape[i]))
        self.biases = []
        for i in range(len(shape)-1):
            self.biases.append(np.random.rand(shape[i+1]))

    def getLayer(self, layer):
        return self.weights[layer], self.biases[layer]
    
    def sigmoid(self, vector):
        for i in range(len(vector)):
            vector[i] = 1.0 / (1.0 + math.exp(-vector[i]))
        return vector
    
    def forward(self, x):
        w0 = self.weights[0]
        b0 = self.biases[0]
        v = self.sigmoid(vector=np.add(np.dot(w0, x), b0))
        for w, b in zip(self.weights[1:-1], self.biases[1:-1]):
            v = self.sigmoid(np.add(np.dot(w, v), b))
        output =np.add(np.dot(self.weights[-1], v), self.biases[-1])
        return output
        
        
"""        
n = NeuralNetworks(shape=[2,3,4,5,1])
for i in range(len(n.weights)):
    print(n.weights[i].shape, n.biases[i].shape)

print(n.forward(np.array([1.0,1.0])))
"""
