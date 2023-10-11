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
fig3, axes3 = plt.subplots(len(files), 3, figsize=(12, 6 * len(files)))

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
    axes[i, 0].imshow(tif_float, cmap='gray')  # No need to convert to uint8
    axes[i, 0].set_title(f'Original {i + 1}')
    axes[i, 0].axis('off')

    # Apply average filter to the image
    lap1_image = func.laplacefilter1(tif_float)
    axes[i, 1].imshow(lap1_image, cmap='gray')  # No need to convert to uint8
    axes[i, 1].set_title(f'laplace 1 {i + 1}')
    axes[i, 1].axis('off')

    # Apply gaussian filter
    lap2_image = func.laplacefilter2(tif_float)
    axes[i, 2].imshow(lap2_image, cmap='gray')  # No need to convert to uint8
    axes[i, 2].set_title(f'laplace 2 {i + 1}')
    axes[i, 2].axis('off')

    # Original image
    axes2[i, 0].imshow(tif_float, cmap='gray')  # No need to convert to uint8
    axes2[i, 0].set_title(f'Original {i + 1}')
    axes2[i, 0].axis('off')

    # sharpen with first filter
    sharp1 = tif_float + 0.5*lap1_image
    axes2[i, 1].imshow(sharp1, cmap='gray')  # No need to convert to uint8
    axes2[i, 1].set_title(f'sharpened 1 {i + 1}')
    axes2[i, 1].axis('off')

    # sharpen with second filter
    sharp2 = tif_float + 0.5*lap2_image
    axes2[i, 2].imshow(sharp2, cmap='gray')  # No need to convert to uint8
    axes2[i, 2].set_title(f'sharpened 2 {i + 1}')
    axes2[i, 2].axis('off')

    # Original image
    axes3[i, 0].imshow(tif_float, cmap='gray')  # No need to convert to uint8
    axes3[i, 0].set_title(f'Original {i + 1}')
    axes3[i, 0].axis('off')

    # sharpen with high pass
    hipass = tif_float - func.my_average_filter(tif_float, 11)
    sharp3 = tif_float + hipass
    axes3[i, 1].imshow(sharp3, cmap='gray')  # No need to convert to uint8
    axes3[i, 1].set_title(f'high pass sharp {i + 1}')
    axes3[i, 1].axis('off')

    # Gaussian unsharp masking
    blurred_image = gaussian_filter(tif_float, 2)
    sharpened_image = tif_float + (tif_float - blurred_image)
    axes3[i, 2].imshow(sharpened_image, cmap='gray')
    axes3[i, 2].set_title(f'gaussian sharp {i + 1}')
    axes3[i, 2].axis('off')

fig.suptitle("Images", fontsize=16)
plt.show()