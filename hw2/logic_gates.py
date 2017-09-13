from neural_network import NeuralNetwork
import numpy as np

class AND(NeuralNetwork):
    def __init__(self):
        NeuralNetwork.__init__(self, shape=[2,2,1])
        self.weights[0] = np.array([[5,5],[5,5]], dtype=np.float32)
        self.biases[0] = np.array([-10, -10], dtype=np.float32)
        self.weights[1] = np.array([1, 1], dtype=np.float32)
        self.biases[1] = np.array([0], dtype=np.float32)

    def forward(self, x=bool(), y=bool()):
        x = np.array([x, y], dtype=np.float32)
        output = NeuralNetwork.forward(self, x)
        if output > 0.9:
            return True
        if output < 0.1:
            return False



# class NOT()

# class OR()

# class XOR()

g = AND()
print(g.forward(False, False))
