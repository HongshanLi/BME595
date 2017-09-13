import math

def myfunc(**kwargs):
    for k, v in kwargs.items():
        print (k, v)

def sigmoid(vector):
    for i in range(len(vector)):
        vector[i] = 1.0 / (1 + math.exp(-vector[i]))
    return vector

print(sigmoid([0,0,0,0]))
