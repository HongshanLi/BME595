from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from conv import Conv2D

# load the image, convert it to a numpy array, and normalize it
def image_to_numpy(image):
    img = Image.open(image)
    img = np.asarray(img)
    img = img.astype(float)
    maximum_pixel = max(img.ravel())
    img = img / maximum_pixel # normalize each pixel to [0, 1]
    img = np.transpose(img, (2, 0, 1))
    return img


def initialize_the_filter(Conv2D, K):
    """
    Arguments:
        K: a list of initial values
    """
    for i in range(Conv2D.in_channels):
        for j in range(Conv2D.o_channels):
            Conv2D.filter[i][j]=K[j]
    return


def save_each_channel(output, image_name):
    o_channels = output.shape[0]
    for i in range(o_channels):
        output_name = image_name + str(i) + ".jpg"
        channel = output[i]
        channel = (channel - channel.min()) / (channel.max() - channel.min())
        channel = channel * 255.9
        channel = channel.astype(np.uint8)
        image = Image.fromarray(channel)
        image.save(output_name)

def part_A():
    task1 = [np.array([[-1,-1,-1],[0,0,0],[1,1,1]], dtype=np.float64)]
    task2 = [np.array([[-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1],[0,0,0,0,0],[1,1,1,1,1],[1,1,1,1,1]], 
             dtype=np.float64),
             np.array([[-1,-1,0,1,1],[-1,-1,0,1,1],[-1,-1,0,1,1],[-1,-1,0,1,1],[-1,-1,0,1,1]], 
             dtype=np.float64)]
    task3 = [np.array([[-1,-1,-1],[0,0,0],[1,1,1]], dtype=np.float64),
               np.array([[-1,0,1],[-1,0,1],[-1,0,1]], dtype=np.float64),
               np.array([[1,1,1],[1,1,1],[1,1,1]], dtype=np.float64)]
    
    images = ["mountain.jpg", "city.jpg"]
    
    # Task 1
    for image in images:
        img = image_to_numpy(image)
        conv = Conv2D(in_channels=3, o_channels=1, kernel_size=3, stride=1, padding=True)
        initialize_the_filter(conv, task1)
        output = conv.forward(img)
        image_name=image.split(".")[0] + "Task1"
        save_each_channel(output, image_name)

    # Task 2
    for image in images:
        img = image_to_numpy(image)
        conv = Conv2D(in_channels=3, o_channels=2, kernel_size=5, stride=1, padding=True)
        initialize_the_filter(conv, task2)
        output = conv.forward(img)
        image_name = image.split(".")[0] + "Task2"
        save_each_channel(output, image_name)
                                        
    # Task 3
    for image in images:
        img = image_to_numpy(image)
        conv = Conv2D(in_channels=3, o_channels=3, kernel_size=3, stride=2, padding=True)
        initialize_the_filter(conv, task3)
        output = conv.forward(img)
        image_name=image.split(".")[0] + "Task3"
        save_each_channel(output, image_name)
    return 

def part_B():
    # set o_channels = 2^i, and plot the time for executing the convolution against i
    from time import time
    
    img = image_to_numpy("mountain.jpg")
    run_time = []
    log_2_o_channels = []
    for i in range(11):
        o_channels = 2**i
        start = time()
        conv = Conv2D(in_channels=3, o_channels=o_channels, kernel_size=3, stride=1, padding=True)
        result = conv.forward(img)
        finish = time()
        duration = finish - start
        run_time.append(duration)
        log_2_o_channels.append(i)

    plt.scatter(x=log_2_o_channels, y=run_time)
    plt.xlabel("log_2_o_channels")
    plt.ylabel("run_time")
    plt.savefig("RunTimeVsOutputChannels.jpg")        
    return

def main():
    part_A()
    part_B()
    return 
   
if __name__=="__main__":
    main()

