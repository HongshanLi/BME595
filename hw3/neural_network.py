import numpy as np
import math
from six.moves import xrange

class NeuralNetwork():
    def __init__(self, shape=list()):
        self.shape = shape
        
        self.layers = []
        for i in range(len(self.shape)): # output of each layer
            self.layers.append(np.zeros([1, self.shape[i]]))

        self.Theta = [] # weight of each layer
        for i in range(len(shape)-1):
            self.Theta.append(np.random.rand(shape[i], shape[i+1]))
        
        self.Biases = [] # bias of each layer
        for i in range(len(shape)-1):
            self.Biases.append(np.random.rand(shape[i+1]))
        
        
        self.dlayer_dTheta = []
        i = 0
        for layer in self.layers[1:]:
            self.dlayer_dTheta.append(np.zeros([1, self.shape[i], self.shape[i+1]]))
            i+=1
         
        self.local_derivative = []
        i=0
        for layer in self.layers[1:]:
            self.local_derivative.append(np.zeros([1, shape[i], shape[i+1]]))
            i+=1

        self.dE_dTheta = []
        i=0
        for layer in self.layers[1:]:
            self.dE_dTheta.append(np.zeros([1, shape[i], shape[i+1]]))
            i+=1
                

        self.dE_dBiases = []
        i=0
        for layer in self.layers[1:]:
            self.dE_dBiases.append(np.zeros([1, shape[i+1]]))
            i+=1

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

        # update local derivative
        def vectorize_each_layer(array, times):
            A = np.concatenate([array]*times, axis=1)
            A = A.reshape(array.shape[0], times, array.shape[1])
            return A

        def vectorize_weight(array, batch_size):
            A = np.concatenate([array]*batch_size, axis=0)
            A = A.reshape(batch_size, array.shape[0], array.shape[1])
            return A
        

        batch_size = self.layers[0].shape[0]
        i=0
        for weight, layer in zip(self.Theta, self.layers[1:]):
            times = self.layers[i].shape[-1]
            L = vectorize_each_layer(layer, times)
            W = vectorize_weight(weight, batch_size)
            self.local_derivative[i] = np.multiply(L, W)
            i+=1

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
        
        # update the local derivative
        i = 0
        for l in self.layers[1:-1]:
            times = self.shape[i]
            L = np.concatenate([l]*times, axis=0)
            self.local_derivative[i] = np.multiply(np.multiply(L, L-1), -self.Theta[i])
            i+=1
        
        # update the last layer of local derivative
        times = self.shape[-2]
        l = absolute_error
        L = np.concatenate([l]*times, axis=0)
        self.local_derivative[-1] = np.multiply(np.multiply(L, L-1), -self.Theta[-1])


        # Chain rule
        for i in range(len(self.dE_dTheta)-1):
            w = Theta_local_gradient[i]
            L = self.local_derivative[i+1:]
            x = L[0]
            for b in L[1:]:
                x = np.dot(x, b)
            x = x.flatten()
            times = w.shape[0]
            x = np.concatenate([x]*times, )
            self.dE_dTheta[i] = np.multiply(w, x)

        self.dE_dTheta[-1] = Theta_local_gradient[-1]





