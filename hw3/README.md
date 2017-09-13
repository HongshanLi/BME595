### The NeuralNetwork Class
To make an instance of a feedforward neural network with n layers,
parse a list of n elements to the instance, each element denotes
the number of neurons in the corresponding layer. For example,
to make an instance of a neural net with 3 layers, such that in=4, h1=2,
out = 1, do the following
```
from neural_network import NeuralNetworks

net = NeuralNetwork([4, 2, 1])
```
You can see and initialize the weight and bias of any layer. Here is an 
example.
To see the weight and bias of the first layer, do
```
net.getLayer(0)
```
To intialize the weight and bias of the first layer with your favorite 
matrices, do
```
net.weights[0] = a numpy array of dimension (2, 4)
net.biases[0] = a numpy array of dimension (2)
```
To forward propogate a 1-d vector,
```
v = numpy 1d vector
net.forward(v)
```

### The Logic Gates
To test the AND gate 
```
from logic_gate import AND
and_ = AND()
and_(True, False)
```

