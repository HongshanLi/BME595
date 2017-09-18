import numpy as np
from neural_network import NeuralNetwork

# I want to see if I have the correct initializaiton
n = NeuralNetwork([2,3,4,1])
print("initial layers")
print(n.layers, "\n")
print("initial weights")
print(n.Theta, "\n")
print("initial dlayer_dTheta")
print(n.dlayer_dTheta, "\n")
print("inital local derivative")
print(n.local_derivative,  "\n")
print("initial dE_dTheta")
print(n.dE_dTheta, "\n")
x = np.array([[2,3],[4,5],[7,8],[9,10],[2,7]], np.float32)
y = np.array([[1000],[2000],[3000],[4000],[5000]], np.float32)
print("forward and backward")
n.forward(x)
print("chain at each level")
print(n.backward(y))





print("updated layers")
print(n.layers)
print("updated weights")
print(n.Theta)
print("updated dlayer_dTheta")
print(n.dlayer_dTheta)
print("updated local derivative")
print(n.local_derivative)
print("updated dE_dTheta")
print(n.dE_dTheta)





"""
WC = np.random.rand(4, 2)
D = np.array([[3, 4]])
S = np.concatenate([D, D, D, D], axis=0)

dD_dC = np.multiply(np.multiply(S, S-1), -WC)

C = np.array([[2,3,4,5]])


print(dD_dC)


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
x = np.array([[2,3],[3,5]])
n.forward(x)
print("output of each layers")
print(n.layers)
print("\n")
print("local biases gradient")
print(n.backward(np.array([[100], [120]], np.float32)))
print("\n")
print("global gradients")
print(n.dE_dTheta)
print("\n")




a = np.array([[1,2],[3,4]])
b = np.array([[1],[10]])
print(b.sum(axis=1))
print(np.multiply(a, b.sum(axis=1)))

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


A = np.arange(1, 13)
A = A.reshape(-1, 2, 3)
B = np.average(A, axis=0) 

print(B)

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


Given array([b1,...,bn]),
want array[b1,,,b1, b2,,,,b2, bn,,,bn]
"""
"""
a = np.array([[1, 2, 3, 4, 5]], dtype=np.float32)
b = np.array([[1,2,3,4]], dtype=np.float32)

a = vectorization(a, 1, 0)
print(a
print("\n")
b = vectorization(b, 5, 1)
print(b)
c = np.multiply(a, b) 

c = c.reshape(5,4)
print(c)
"""
