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


def my_otsu(image):

    # dummy variables, replace with your own code:
    threshold = 128
    between_class_variance = 0
    separability = 0
    # until here

    binary_image = image >= threshold
    return binary_image, between_class_variance, threshold, separability


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
    plt.show()

    plt.imshow(binary_image.astype(int), cmap="gray")
    plt.show()


if __name__ == "__main__":
    main()

