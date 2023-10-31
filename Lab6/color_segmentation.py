from pathlib import Path
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from PIL import Image
from skimage.morphology import binary_erosion, binary_dilation, disk, square
from skimage.segmentation import clear_border
from math import acos, sqrt

# Load the images
file_name = Path('.', 'pics', 'brainCells.tif')
cells = Image.open(file_name)

# extract RGB-channels
rgb_cells = cells.convert('RGB')
R = np.array(rgb_cells)[:,:,0]
G = np.array(rgb_cells)[:,:,1]
B = np.array(rgb_cells)[:,:,2]

# create the CMY channels
C = 255 - R
M = 255 - G
Y = 255 - B

# create HSI channels
I = 1/3 * (R/255 + G/255 + B/255)

H = np.zeros_like(R, dtype=float)
S = np.zeros_like(R, dtype=float)

for i in range(R.shape[0]):
    for j in range(R.shape[1]):
        numerator = 0.5 * ((R[i, j]/255 - G[i, j]/255) + (R[i, j]/255 - B[i, j]/255))
        denominator = sqrt((R[i, j]/255 - G[i, j]/255) ** 2 + (R[i, j]/255 - B[i, j]/255) * (G[i, j]/255 - B[i, j]/255))
        
        if denominator == 0:
            H[i, j] = 0
        else:
            H[i, j] = acos(numerator / denominator)

        if B[i, j] > G[i, j]:
            H[i, j] = 2*np.pi - H[i, j]

        # Calculate Saturation (S)
        min_rgb = min(R[i, j], G[i, j], B[i, j]) / 255
        S[i, j] = 1 - (3 * min_rgb / (R[i, j]/255 + G[i, j]/255 + B[i, j]/255))

# Scale the Hue channel to the range [0, 255]
scaled_H = (H * 255 / (2 * np.pi)).astype(np.uint8)

# Define the lower and upper bounds
lower_bound = 130
upper_bound = 170

# Define custom colormaps for each image with specified boundaries
colors1 = [(0.2, 0, 1), (1, 0, 0), (0.2, 0, 1)]
boundaries1 = [0, lower_bound, upper_bound, 255]
cmap1 = ListedColormap(colors1)
norm1 = BoundaryNorm(boundaries1, cmap1.N, clip=True)

# Create a binary mask for the image
binary_mask = np.logical_and(scaled_H >= lower_bound, scaled_H <= upper_bound)
binary_image = binary_mask.astype(np.uint8)

# get rid of noise
eroded_image = binary_erosion(binary_image, disk(2))
dilated_image = binary_dilation(eroded_image, disk(2))
eroded_image = binary_erosion(dilated_image, square(2))
dilated_image = binary_dilation(eroded_image, square(2))

# clear borders
clear_image = clear_border(dilated_image)

# label and count the segments
cells_segments, num_segments = ndimage.label(clear_image)
print('found segments: ',num_segments)

# ----------------------------------- plots -------------------------------------------------
fig1, axes1 = plt.subplots(3, 3, figsize=(12, 8))
# Plot the RGB-channels
axes1[0,0].imshow(R, cmap='gray')
axes1[0,0].set_title('R')
axes1[0,0].axis('off')
axes1[0,1].imshow(G, cmap='gray')
axes1[0,1].set_title('G')
axes1[0,1].axis('off')
axes1[0,2].imshow(B, cmap='gray')
axes1[0,2].set_title('B')
axes1[0,2].axis('off')
# Plot the CMY-channels
axes1[1,0].imshow(C, cmap='gray')
axes1[1,0].set_title('C')
axes1[1,0].axis('off')
axes1[1,1].imshow(M, cmap='gray')
axes1[1,1].set_title('M')
axes1[1,1].axis('off')
axes1[1,2].imshow(Y, cmap='gray')
axes1[1,2].set_title('Y')
axes1[1,2].axis('off')
# Plot the HSI-channels
axes1[2,0].imshow(H, cmap='gray')
axes1[2,0].set_title('H')
axes1[2,0].axis('off')
axes1[2,1].imshow(S, cmap='gray')
axes1[2,1].set_title('S')
axes1[2,1].axis('off')
axes1[2,2].imshow(I, cmap='gray')
axes1[2,2].set_title('I')
axes1[2,2].axis('off')
plt.show()

fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
# Plot Hue channel
axes2[0,0].imshow(scaled_H, cmap='gray')
axes2[0,0].set_title('Hue')
axes2[0,0].axis('off') 
# Add a subplot for the histogram in the second row
axes2[1,0].hist(scaled_H.ravel(), bins=256, range=(0, 256), density=True, color='b', alpha=0.7)
axes2[1,0].set_title('Hue Histogram')
axes2[1,0].set_xlabel('Hue Value')
axes2[1,0].set_ylabel('Normalized Frequency')
axes2[1,0].set_xlim(0, 255)
# Plot Hue channel
axes2[0,1].imshow(scaled_H, cmap=cmap1, norm=norm1)
axes2[0,1].set_title('custom cmap')
axes2[0,1].axis('off') 
# Plot Hue channel
axes2[1,1].imshow(binary_image, cmap='gray')
axes2[1,1].set_title('binary image')
axes2[1,1].axis('off') 
plt.show()

fig3, axes3 = plt.subplots(1, 2, figsize=(12, 8))
# binary image
axes3[0].imshow(binary_image, cmap='gray')
axes3[0].set_title('binary image')
axes3[0].axis('off')
# clean image
axes3[1].imshow(clear_image, cmap='gray')
axes3[1].set_title('cleaned up image')
axes3[1].axis('off')
plt.show()