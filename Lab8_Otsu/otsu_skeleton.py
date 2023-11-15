import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image

def basic_thresholding(image):
    imnew = image.flatten()
    t = np.mean(image).astype(int)
    while True:
        seg_a = imnew[imnew > t]
        seg_b = imnew[imnew <= t]
        t_a = np.mean(seg_a)
        t_b = np.mean(seg_b)
        new_t = np.round((t_a + t_b) / 2)

        if np.abs(t - new_t) < 0.1:
            break

        t = new_t

    bin_img = image > t
    return bin_img, t


import numpy as np

def my_otsu(image):
    # Calculate the histogram of the input image
    hist, bins = np.histogram(image, bins=256, range=(0, 256))
    
    # Compute total number of pixels
    total_pixels = np.sum(hist)
    # total mean value
    m_G = np.dot(np.arange(256), hist) / total_pixels
    # array for between class variances
    between_class_array = np.empty(256)
    
    # initializations
    between_class_variance_max = 0
    threshold = 0
    separability = 0
    G_variance = 0

    for k in range(1, 256):
        class_a = hist[:k]
        class_b = hist[k:]
        
        sum_a = np.sum(class_a)
        sum_b = np.sum(class_b)
        
        # Compute the probabilities of each class
        prob_a = sum_a / total_pixels
        prob_b = sum_b / total_pixels
        
        # Calculate the mean of each class
        m_a = np.dot(np.arange(k), class_a) / sum_a if sum_a > 0 else 0
        m_b = np.dot(np.arange(k, 256), class_b) / sum_b if sum_b > 0 else 0
        
        # Calculate between-class variance
        between_class_variance = prob_a * (m_a - m_G) ** 2 + prob_b * (m_b - m_G) ** 2
        G_variance += hist[k]*(k - m_G) ** 2
        between_class_array[k] = between_class_variance

        if between_class_variance > between_class_variance_max:
            between_class_variance_max = between_class_variance
            threshold = k

    # Apply the threshold to the image
    binary_image = image >= threshold
    separability = between_class_variance_max / G_variance

    return binary_image, between_class_array, threshold, separability




def main():
    #image = np.asarray(Image.open("polymersome_cells_10_36.png"))
    image = np.asarray(Image.open("thGonz.tif"))

    #image = np.asarray(Image.open("binary_test_image.png"))
    if len(image.shape)==3:
        image=image[:,:,0]

    binary_image, t = basic_thresholding(image)
    print("Basic Thresholding. Output Threshold: "+str(t))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(binary_image.astype(int), cmap="gray")
    plt.title("basic thresholding")
    plt.subplot(1, 2, 2)
    plt.hist(image.ravel(), bins=256)
    plt.axvline(t, color='r', linestyle='dashed', linewidth=2)
    plt.show()

    binary_image, between_class_variance, threshold, separability = my_otsu(image)

    print("Otsu's Method. Output Threshold: "+str(threshold))
    print("Separability: "+str(separability))
    #print(separability)

    plt.plot(between_class_variance*100)
    plt.title("between class variance")
    plt.show()

    plt.imshow(binary_image.astype(int), cmap="gray")
    plt.show()


if __name__ == "__main__":
    main()

