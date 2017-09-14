import numpy as np
import math
from six.moves import xrange

class NeuralNetwork():
    def __init__(self, shape=list()):
        self.shape = shape
        self.layers = [np.zeros([self.shape[i]]).reshape(-1,self.shape[i]) 
            for i in xrange(len(self.shape))] # output of each layer
        self.Theta = [] # weight of each layer
        for i in range(len(shape)-1):
            self.Theta.append(np.random.rand(shape[i], shape[i+1]))
        
        self.Biases = [] # bias of each layer
        for i in range(len(shape)-1):
            self.Biases.append(np.random.rand(shape[i+1]))

        self.dE_dTheta = [0 for i in range(len(self.Theta))] # Gradient of weight of each layer

        self.dE_dBiases = [0 for i in range(len(self.Biases))] # Gradient of biases of each layer

    def getLayer(self, layer):
        return self.weights[layer], self.biases[layer]
    
    def forward(self, x):
        def sigmoid(array):
            return 1.0 / (1.0 + np.exp(array))
        
        dim = x.shape[-1]
        x = x.reshape(-1, dim)
        self.layers[0] = x
        i = 1
        for w, b in zip(self.Theta[:-1], self.Biases[:-1]):
            x = sigmoid(np.dot(x, w) + b)
            self.layers[i] = x
            i+=1

        w = self.Theta[-1]
        b = self.Biases[-1]
        x = np.dot(x, w) + b
        self.layers[-1] = x
        return x

    def backward(self, target):
        absolute_error = np.abs(self.layers[-1] - target)
        
        def vectorize_left_layer(array, times):
            array = array.flatten("C")
            array = array.reshape(1, -1)
            out = array
            for i in range(1,times):
                out = np.concatenate([out, array], axis=0)
            out = out.flatten("F")
            return out 

        def vectorize_right_layer(array, times):
            out = array
            for i in range(1, times):
                out = np.concatenate([out, array], axis=1)
            out = out.flatten("C")
            return out
        

        Theta_local_gradient = [0 for i in range(len(self.Theta))]
        i = 0
        for w in Theta_local_gradient[:-1]:
            w_right = self.layers[i+1]
            w_left = self.layers[i]
            right_time = w_left.shape[-1]
            left_time = w_right.shape[-1]
            w_right = vectorize_right_layer(w_right, right_time)
            sigmoid_prime = w_right*(w_right - 1)
            w_left = vectorize_left_layer(w_left, left_time)
            w = np.multiply(w_left, sigmoid_prime)
  
            w = w.reshape(-1, right_time, left_time)
            Theta_local_gradient[i] = w
            i+=1 
        # To compute the local gradient of the last layer, we need backprop abosolute_error
        w_right = absolute_error
        w_left = self.layers[-2]
        right_time = w_left.shape[-1]
        left_time = w_right.shape[-1]
        w_right = vectorize_right_layer(w_right, right_time)
        w_left = vectorize_left_layer(w_left, left_time)
        w = np.multiply(w_left, w_right)
        w = w.reshape(-1, right_time, left_time)
        Theta_local_gradient[-1] = w

        # Multiply local gradients together to get global gradient by chain rule
        # 
        self.dE_dTheta = Theta_local_gradient
        
        for i in range(1, len(self.dE_dTheta)):        
            A = self.dE_dTheta[-i] 
            B = self.dE_dTheta[-(i+1)]
            num_neuron = B.shape[1]
            A = A.sum(axis=-1)
            A = vectorize_left_layer(A, num_neuron)
            A = A.reshape(B.shape[0], B.shape[1], B.shape[2])
            self.dE_dTheta[-(i+1)] = np.multiply(B, A)


        return Theta_local_gradient
        
         
     
