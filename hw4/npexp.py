import numpy as np

def soft_max(output_layer):
    D = np.exp(output_layer).sum(axis=1)
    times = output_layer.shape[1]
    D = np.concatenate([D]*times, axis=0)
    D = D.flatten("F")
    D = D.reshape(output_layer.shape[0], output_layer.shape[1])
    N = np.exp(output_layer)
    
    return N / D

A = np.ones([5,5])
print(A)
print(soft_max(A))

