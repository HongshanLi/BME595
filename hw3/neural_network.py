import numpy as np
import math
from six.moves import xrange

class NeuralNetwork():
    def __init__(self, shape=list()):
        self.shape = shape
        
        self.layers = []
        for i in range(len(self.shape)):
            self.layers.append(np.zeros([1, self.shape[i]]))

        self.Theta = [] # weight of each layer
        for i in range(len(shape)-1):
            self.Theta.append(np.random.rand(shape[i], shape[i+1]))
        
        self.dlayer_dTheta = [] 
        i = 0
        for layer in self.layers[1:]:
            self.dlayer_dTheta.append(np.zeros([1, self.shape[i], self.shape[i+1]]))
            i+=1
         
        self.local_derivative = [] # dlayer[n] /dlayer[n-1]
        i=0
        for layer in self.layers[1:]:
            self.local_derivative.append(np.zeros([1, shape[i], shape[i+1]]))
            i+=1

        self.dE_dTheta = []
        i=0
        for layer in self.layers[1:]:
            self.dE_dTheta.append(np.zeros([shape[i], shape[i+1]]))
            i+=1
                

    
    def forward(self, x):
        num_layers = len(self.shape)
        
        def sigmoid(array):
            return 1.0 / (1.0 + np.exp(array))
        
        x = x.reshape(-1, self.shape[0])
        self.layers[0] = x
        i = 1
        for w in self.Theta:
            if i < num_layers - 1:
                x = sigmoid(np.dot(x, w))
            if i == num_layers -1:    
                x = np.dot(x, w)
            self.layers[i] = x
            i+=1

    


    def backward(self, target):
        absolute_error = np.abs(self.layers[-1] - target)
        self.layers[-1] = absolute_error
        
        def vectorize_each_layer(array, times):
            A = np.concatenate([array]*times, axis=1)
            A = A.reshape(array.shape[0], times, array.shape[1])
            return A

        def vectorize_weight(array, batch_size):
            A = np.concatenate([array]*batch_size, axis=0)
            A = A.reshape(batch_size, array.shape[0], array.shape[1])
            return A
        
        # Update local derivates
        batch_size = self.layers[0].shape[0]
        num_layers = len(self.shape)

        i=0
        for weight, layer in zip(self.Theta, self.layers[1:]):
            times = self.layers[i].shape[-1]
            L = vectorize_each_layer(layer, times)
            W = vectorize_weight(weight, batch_size)
            
            if i < num_layers - 2:
                C = np.multiply(np.multiply(L, L-1), -W).mean(axis=0)
            else:
                C = np.multiply(L, W).mean(axis=0) # There is no nonlinearity for the output
            
            self.local_derivative[i] = C
            i+=1

        
        # Update dlayer_dTheta
        def vectorize_input(array, batch_size, next_layer_size):
            A = array.flatten("C")
            A = A.reshape(1, -1)
            A = np.concatenate([A]*next_layer_size, axis=0)
            A = A.flatten("F")
            A = A.reshape(batch_size, array.shape[1], next_layer_size )
            return A
        
        i=0
        for layer, next_layer in zip(self.layers[:-1], self.layers[1:]):
            A = vectorize_input(layer, batch_size, next_layer.shape[-1])
            B = vectorize_each_layer(next_layer, times=layer.shape[-1])
            if i < num_layers -2:
                C = np.multiply(np.multiply(B, B-1), -A).mean(axis=0)
                self.dlayer_dTheta[i] = C
            else:
                C = np.multiply(B, A).mean(axis=0)
            
            self.dlayer_dTheta[i] = C
            i+=1


        # Compute dE_dTheta via chain rule
        def compute_chain(dlayer_dTheta_index):
            # it computes the chain after dlayer_dTheta[dlayer_dTheta_index]
            j = dlayer_dTheta_index + 1
            x = self.local_derivative[j]
            for y in self.local_derivative[j+1:]:
                x = np.dot(x, y)
            return x
        
        def vectorize_chain(array, times):
            A = array.flatten("C")
            A = A.reshape(1,array.shape[0])
            A = np.concatenate([A]*times, axis=0)
            return A

        
        tmp1 = []
        tmp2 = []
        i = 0
        for dw in self.dlayer_dTheta:
            if i < num_layers - 2:
                times = dw.shape[0]
                C = compute_chain(i)
                C = vectorize_chain(C, times)
                self.dE_dTheta[i] = np.multiply(dw, C)
            else:
                self.dE_dTheta[i] = dw
            i+=1

        
            
            
        return tmp1, tmp2





