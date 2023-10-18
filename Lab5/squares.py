from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image

# ----------------------------------------------------- load image ---------------------------------------------
file_name = Path('.', 'pics', 'squares.tif')

# Open the image with pillow and convert to numpy array
image = Image.open(file_name)
bin_image = np.asarray(Image.open(file_name))

nRows = bin_image.shape[0]
nCols = bin_image.shape[1]

ker5 = np.ones((5, 5), dtype=bool)
ker6 = np.ones((6, 6), dtype=bool)

eroded5 = ndimage.binary_erosion(bin_image, structure=ker5)
im5 = ndimage.binary_dilation(eroded5, structure=ker5)
eroded6 = ndimage.binary_erosion(bin_image, structure=ker6)
im6 = ndimage.binary_dilation(eroded6, structure=ker6)

only5 = np.bitwise_xor(im5,im6)

# Iterate over the entire image and leave the loop empty
segments = []
input_image = only5
while np.sum(input_image) > 0:
    search = True
    segment = np.zeros((nRows, nCols), dtype=bool)
    for i in range(nRows):
        for j in range(nCols):
            if(input_image[i,j] and search):
                search = False
                segment[i,j] = 1
            if(input_image[i,j] and segment[i-1,j]
            or input_image[i,j] and segment[i+1,j]
            or input_image[i,j] and segment[i,j-1]
            or input_image[i,j] and segment[i,j+1]):
                segment[i,j] = 1
    segments.append(segment)
    input_image = np.bitwise_xor(input_image,segment)

print (len(segments))

# --------------------------------------------------------- display images ---------------------------------------------
fig0 = plt.figure(1)
plt.subplot(2, 2, 1)
plt.title('original')
plt.imshow(bin_image, cmap='gray')

plt.subplot(2, 2, 2)
plt.title('>=5')
plt.imshow(im5, cmap='gray')

plt.subplot(2, 2, 3)
plt.title('>=6')
plt.imshow(im6, cmap='gray')

plt.subplot(2, 2, 4)
plt.title('result')
plt.imshow(only5, cmap='gray')

plt.tight_layout()
for ax in fig0.get_axes():
    ax.axis('off')
plt.show()

# ------------------------------- segments ------------------------------
fig1 = plt.figure(2)
plt.subplot(2, 2, 1)
plt.title('original')
plt.imshow(segments[0], cmap='gray')

plt.subplot(2, 2, 2)
plt.title('original')
plt.imshow(segments[1], cmap='gray')

plt.subplot(2, 2, 3)
plt.title('original')
plt.imshow(segments[2], cmap='gray')

plt.subplot(2, 2, 4)
plt.title('original')
plt.imshow(segments[3], cmap='gray')

plt.tight_layout()
for ax in fig1.get_axes():
    ax.axis('off')
plt.show()