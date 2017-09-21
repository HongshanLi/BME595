import torchvision.datasets as dset
import numpy as np

mnist = dset.MNIST(root="/homes/li108/Dataset/Mnist", train=True)
im, label = mnist[2]
pix = np.array(im)
print(pix)
print(pix.shape)
