import numpy as np
import func
import matplotlib.pyplot as plt
from skimage import io, color
from scipy.ndimage import gaussian_filter
import os

# Directory containing your DICOM images
imdir = './tifs/'

# List all tif files in the directory and create an array
files = [os.path.join(imdir, fname) for fname in os.listdir(imdir) if fname.endswith('.tif')]

# Create a figure for the subplots
fig, axes = plt.subplots(len(files), 3, figsize=(12, 6 * len(files)))
fig2, axes2 = plt.subplots(len(files), 3, figsize=(12, 6 * len(files)))

# Loop through and display all DICOM images
for i, file_path in enumerate(files):
    # Open the TIFF file
    tif_image = io.imread(file_path)

    # Convert the TIFF image to grayscale (if not already)
    if tif_image.ndim == 3:  # Check if the image has multiple channels
        tif_gray = color.rgb2gray(tif_image)
    else:
        tif_gray = tif_image

    # Scale grayscale values to floats between 0 and 1
    tif_float = tif_gray.astype(np.float32) / np.max(tif_gray)
    axes[i, 0].imshow((tif_float * 255).astype(np.uint8), cmap='gray')  # Convert to uint8
    axes[i, 0].set_title(f'Original {i + 1}')
    axes[i, 0].axis('off')

    # Apply average filter to the image
    avg_image = func.my_average_filter(tif_float, 11)
    axes[i, 1].imshow((avg_image * 255).astype(np.uint8), cmap='gray')  # Convert to uint8
    axes[i, 1].set_title(f'average filter {i + 1}')
    axes[i, 1].axis('off')

    # Apply gaussian filter
    gauss_image = func.my_gauss_filter(tif_float, 11, 0.85)
    axes[i, 2].imshow((gauss_image * 255).astype(np.uint8), cmap='gray')  # Convert to uint8
    axes[i, 2].set_title(f'gauss filter {i + 1}')
    axes[i, 2].axis('off')

    # Original image
    axes2[i, 0].imshow((tif_float * 255).astype(np.uint8), cmap='gray')  # Convert to uint8
    axes2[i, 0].set_title(f'Original {i + 1}')
    axes2[i, 0].axis('off')

    # High pass filter with average filter subtract
    hipass1 = (tif_float - avg_image) * 255
    axes2[i, 1].imshow(hipass1.astype(np.uint8), cmap='gray')  # Convert to uint8
    axes2[i, 1].set_title(f'high pass average {i + 1}')
    axes2[i, 1].axis('off')

    # High pass filter with gauss filter subtract
    hipass2 = (tif_float - gauss_image) * 255
    axes2[i, 2].imshow(hipass2.astype(np.uint8), cmap='gray')  # Convert to uint8
    axes2[i, 2].set_title(f'high pass gauss {i + 1}')
    axes2[i, 2].axis('off')

fig.suptitle("Images", fontsize=16)
plt.show()