import numpy as np

def pad_the_channel(A, kernel_size):
    pad_size = kernel_size - 1
    pad = np.zeros((A.shape[0], pad_size)) # vertical pad
    A = np.concatenate([A, pad], axis=1)
    pad = np.zeros((pad_size, A.shape[1]))
    A = np.concatenate([A, pad], axis=0) # horizontal pad
    return A
    

def basic_conv2d(A, B, bias, stride, padding=bool):
    """
    Arguments:
        A: a 2d numpy array representing one channel of an image
        B: a smaller 2d square numpy array convolving over A
    Return:
        Convolution of B over A
    """

    kernel_size = B.shape[0]
    
    if padding==True:
        A = pad_the_channel(A, kernel_size)
    [width, length] = A.shape
    
   
    convolution = []
    for i in range(0,width-kernel_size+1,stride):
        horizontal = []
        for j in range(0, length-kernel_size+1, stride):
            sub = A[i:i+kernel_size, j:j+kernel_size]
            conv = np.multiply(sub, B)
            conv = conv.ravel()
            conv = sum(conv) + bias
            horizontal.append(conv)
          
        convolution.append(horizontal)
    
    convolution = np.array(convolution)
    return convolution
        
       

class Conv2D(object):
    def __init__(self,in_channels, o_channels, kernel_size, stride, padding):
        self.in_channels = in_channels
        self.o_channels = o_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.filter = np.random.rand(in_channels, o_channels, kernel_size, kernel_size)
        self.bias_array = np.random.rand(self.in_channels, self.o_channels)

    
    def forward(self, x):
        # x is a 3D tensor (channels, x-axis, y-axis)
        dim = basic_conv2d(A=x[0], B=self.filter[0][0], bias=0, stride=self.stride,
                                padding=self.padding).shape
        output = np.zeros([self.o_channels, dim[0], dim[1]])

        for i in range(self.in_channels):
            for j in range(self.o_channels):
                output[j] = output[j] + basic_conv2d(x[i], self.filter[i][j], stride=self.stride, 
                                                     bias=self.bias_array[i][j], padding=self.padding)
        
        return output

