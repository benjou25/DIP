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

# --------------------- get segment pixel-sizes -----------------------------------
segment_sizes = []
for label in range(1, num_segments + 1):
    # get the current segment
    segment_mask = cells_segments == label
    # Count the number of white pixels in the segment
    num_pixels = np.sum(segment_mask)
    segment_sizes.append(num_pixels)

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

# Create a histogram of segment sizes
plt.hist(segment_sizes, bins=20, color='red', alpha=0.7)
plt.xlabel('Number of Pixels')
plt.ylabel('Frequency')
plt.title('Segment Sizes Histogram')
plt.show()