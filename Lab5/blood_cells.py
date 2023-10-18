from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.segmentation import clear_border
from PIL import Image

# ----------------------------------------------------- load image ---------------------------------------------
# Load the image
file_name = Path('.', 'pics', 'bloodCells.tif')
image = Image.open(file_name)

# Convert the image to grayscale if it's not already
gray_image = image.convert('L')
# Define the threshold value
threshold = 75
# Apply the threshold to create a binary image
binary_image = gray_image.point(lambda p: p > threshold and 255)

# clean up the image
binary_image = np.array(binary_image)
clear_image = clear_border(binary_image)
fill_image = ndimage.binary_fill_holes(clear_image)

# label the segments
cells_segments, num_segments = ndimage.label(fill_image)
print('found segments: ',num_segments)

# --------------------------------------------------------- display images ---------------------------------------------
fig0 = plt.figure(1)
plt.subplot(2, 2, 1)
plt.title('original')
plt.imshow(gray_image, cmap='gray')

plt.subplot(2, 2, 2)
plt.title('binary image')
plt.imshow(binary_image, cmap='gray')

plt.subplot(2, 2, 3)
plt.title('clean image')
plt.imshow(fill_image, cmap='gray')

plt.tight_layout()
for ax in fig0.get_axes():
    ax.axis('off')
plt.show()