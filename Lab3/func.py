import numpy as np
from scipy.signal import convolve2d

def my_average_filter(image, size):
    hm = (1/(size**2))*np.ones([size,size])
    filtered_image = convolve2d(image, hm, mode='same')
    return filtered_image

def my_gauss_filter(image, size, sigma):
    # Create a 2D Gaussian filter kernel
    kernel = np.fromfunction(
        lambda x, y: (1/ (2 * np.pi * sigma ** 2)) * 
                      np.exp(- ((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)),
        (size, size)
    )
    kernel /= kernel.sum()
    # Apply the filter using convolve2d
    filtered_image = convolve2d(image, kernel, mode='same')  
    return filtered_image

def laplacefilter1(image):
    h1 = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
    filtered_image = convolve2d(image, h1, mode='same')
    return filtered_image

def laplacefilter2(image):
    h2 = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
    filtered_image = convolve2d(image, h2, mode='same')
    return filtered_image