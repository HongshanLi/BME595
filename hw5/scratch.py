from img2obj import LeNet
import numpy as np
import torch
img = np.random.rand(3, 32, 32)
n = LeNet()
b = torch.from_numpy(img)

print(n.forward(img))


