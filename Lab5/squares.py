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

# subtract from the squares >5 the squares >6 to get only squraes in shape 5x5
only5 = np.bitwise_xor(im5,im6)

# Iterate over the entire image and detect segments
segment_array = []
input_image = only5
while np.sum(input_image) > 0:
    search = True
    segment = np.zeros((nRows, nCols), dtype=bool)
    for i in range(nRows):
        for j in range(nCols):
            if(input_image[i,j] and search):            # find the seed pixel
                search = False
                segment[i,j] = 1
            if(input_image[i,j] and segment[i-1,j]      # find connected pixels and
            or input_image[i,j] and segment[i+1,j]      # add them to the segment
            or input_image[i,j] and segment[i,j-1]
            or input_image[i,j] and segment[i,j+1]):
                segment[i,j] = 1
    segment_array.append(segment)                       # append the found segment to the array
    input_image = np.bitwise_xor(input_image,segment)   # subtract the detected segment from the input

print ('Found Segments: ',len(segment_array))
print('\nyou could count all white pixels and divide by 5*5 to get the number of 5x5 squares')

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
plt.title('segment 1')
plt.imshow(segment_array[0], cmap='gray')

plt.subplot(2, 2, 2)
plt.title('segment 2')
plt.imshow(segment_array[1], cmap='gray')

plt.subplot(2, 2, 3)
plt.title('segment 3')
plt.imshow(segment_array[2], cmap='gray')

plt.subplot(2, 2, 4)
plt.title('segment 4')
plt.imshow(segment_array[3], cmap='gray')

plt.tight_layout()
for ax in fig1.get_axes():
    ax.axis('off')
plt.show()