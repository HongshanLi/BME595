import numpy as np
from neural_network import NeuralNetwork
"""
n = NeuralNetwork([2, 2, 1])
for w, b in zip(n.weights, n.biases):
    print(w, b, "\n")


print("Start Here")
x = np.array([[2, 2], [3,3],[4,4]])
print(n.forward(x))

print("start here")
x = np.array([[2, 3]])
y = np.array([[1],[1]])
w = np.dot(x, y)
print(w)
print("start here")
x = np.array([[1,2,3,4,5]])
y = np.array([[1, 2, 3]])
z = np.zeros([5,3])
for i in range(5):
    for j in range(3):
        z[i][j] = x[0][i]*y[0][j]

print(z)
"""


"""
Given array([b1,...,bn]),
want array[b1,,,b1, b2,,,,b2, bn,,,bn]
"""
"""
a = np.array([[1, 2, 3, 4, 5]], dtype=np.float32)
b = np.array([[1,2,3,4]], dtype=np.float32)

a = vectorization(a, 1, 0)
print(a
"""
"""
print("\n")
b = vectorization(b, 5, 1)
print(b)
c = np.multiply(a, b) 

c = c.reshape(5,4)
print(c)
"""
n = NeuralNetwork([2,3,4,1])
print("Inital weights")
print(n.Theta)
print("\n")
print("Initial gradient")
print(n.dE_dTheta)
print("\n")
print("Intial Value at each layer")
print(n.layers)
print("feed an array")
x = np.array([[2,3],[5,8],[20, 30], [50, 80]])
n.forward(x)
print("output of each layers")
print(n.layers)
print("\n")
print("local weight gradient")
print(n.backward(np.array([[100], [20], [80], [90]], np.float32)))
print("\n")
print("global gradients")
print(n.dE_dTheta)

"""
a = np.array([[1,2],[3,4]])
b = np.array([[1],[10]])
print(b.sum(axis=1))
print(np.multiply(a, b.sum(axis=1)))
"""
"""
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
    
A = np.array([[1,2,3],[4,5,6]])

print(vectorize_left_layer(A, 3))
print(vectorize_right_layer(A, 3))
"""
"""
A = np.arange(1, 13)
A = A.reshape(2, 6)
A = A.reshape(-1, 2, 3)
print(A)
""" 
