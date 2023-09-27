import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, exposure
import os

# Directory containing your TIFF images
im_dir = './pics/'

# List all TIFF files in the directory and create an array
tif_files = [os.path.join(im_dir, fname) for fname in os.listdir(im_dir) if fname.endswith('.tif')]

# Create a figure for the subplots
fig, axes = plt.subplots(len(tif_files), 4, figsize=(12, 6 * len(tif_files)))

# set gamma value <1 brighten, >1 darken
g_corr = 0.2

# Loop through and display all TIFF images
for i, file_path in enumerate(tif_files):
    try:
        # Open the TIFF file
        tif_image = io.imread(file_path)
        
        # Convert the TIFF image to grayscale (if not already)
        if tif_image.ndim == 3:  # Check if the image has multiple channels
            tif_gray = color.rgb2gray(tif_image)
        else:
            tif_gray = tif_image

        # Scale grayscale values to floats between 0 and 1
        tif_float = tif_gray.astype(np.float32) / np.max(tif_gray)

        # Compute the histogram for the original image
        hist, bins = np.histogram(tif_float, bins=256, range=(0, 1))

        tif_corrected = tif_float ** g_corr

        # Compute the histogram for the corrected image
        hist_new, bins_new = np.histogram(tif_corrected, bins=256, range=(0, 1))

        #use built-in function for gamma correction
        tif_gamma_corrected = exposure.adjust_gamma(tif_gray, gamma=g_corr)

        # Plot the original histogram
        axes[i, 0].hist(bins[:-1], bins, weights=hist)
        axes[i, 0].set_title(f'Original {i + 1}')

        # Plot the corrected histogram
        axes[i, 1].hist(bins_new[:-1], bins_new, weights=hist_new)
        axes[i, 1].set_title(f'Corrected {i + 1}')

        axes[i, 2].imshow(tif_corrected, cmap='gray')
        axes[i, 2].set_title(f'Corrected {i + 1}')
        axes[i, 2].axis('off')

        # Plot the gamma-corrected image using a built-in function
        axes[i, 3].imshow(tif_gamma_corrected, cmap='gray')
        axes[i, 3].set_title(f'Corrected {i + 1}')
        axes[i, 3].axis('off')

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

fig.suptitle("Gamma Correction", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()