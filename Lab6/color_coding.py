from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from PIL import Image

# Load the images
file_name1 = Path('.', 'pics', 'arterie.tif')
pic1 = Image.open(file_name1)
file_name2 = Path('.', 'pics', 'ctSkull.tif')
pic2 = Image.open(file_name2)

# Convert the images to NumPy arrays
pic1_array = np.array(pic1)
pic2_array = np.array(pic2)

# Create histograms
hist1, bins1 = np.histogram(pic1_array, bins=256, range=(0, 256))
hist2, bins2 = np.histogram(pic2_array, bins=256, range=(0, 256))

# Define custom colormaps for each image with specified boundaries
colors1 = [(0, 0, 0.4), (0, 0, 0.6), (0, 0, 0.8), (0, 0, 1), (0, 0.6, 0.9), (0.6, 0, 0), (0.4, 0, 0)]
boundaries1 = [0, 100, 120, 125, 130, 135, 160, 256]
cmap1 = ListedColormap(colors1)
norm1 = BoundaryNorm(boundaries1, cmap1.N, clip=True)

colors2 = [(0, 0, 0.6), (0, 0, 1), (0, 1, 0), (0, 0.6, 0), (0.9, 0, 0), (0.7, 0, 0)]
boundaries2 = [0, 75, 100, 130, 150, 175, 256]
cmap2 = ListedColormap(colors2)
norm2 = BoundaryNorm(boundaries2, cmap2.N, clip=True)

# ------------------- Plot the images, histograms and colormaps -------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# Plot the first image
axes[0, 0].imshow(pic1_array, cmap='gray')
axes[0, 0].set_title('Image 1')
axes[0, 0].axis('off')

# Plot the second image
axes[1, 0].imshow(pic2_array, cmap='gray')
axes[1, 0].set_title('Image 2')
axes[1, 0].axis('off')

# Plot the histogram for Image 1
axes[0, 1].bar(bins1[:-1], hist1, width=1.0, color='blue')
axes[0, 1].set_title('Histogram for Image 1')
axes[0, 1].set_xlabel('Pixel Value')
axes[0, 1].set_ylabel('Frequency')

# Plot the histogram for Image 2
axes[1, 1].bar(bins2[:-1], hist2, width=1.0, color='red')
axes[1, 1].set_title('Histogram for Image 2')
axes[1, 1].set_xlabel('Pixel Value')
axes[1, 1].set_ylabel('Frequency')

# Plot the first image with custom colormap 1
im1 = axes[0, 2].imshow(pic1_array, cmap=cmap1, norm=norm1)
axes[0, 2].set_title('cmap 1')
axes[0, 2].axis('off')
plt.colorbar(im1, ax=axes[0, 2])

# Plot the second image with custom colormap 2
im2 = axes[1, 2].imshow(pic2_array, cmap=cmap2, norm=norm2)
axes[1, 2].set_title('cmap 2')
axes[1, 2].axis('off')
plt.colorbar(im2, ax=axes[1, 2])

plt.tight_layout()
plt.show()