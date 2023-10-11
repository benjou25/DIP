import numpy as np
from scipy.signal import convolve2d

def my_average_filter(image, size):
    hm = (1/(size**2))*np.ones([size,size])
    filtered_image = convolve2d(image, hm, mode='same')
    return filtered_image, hm