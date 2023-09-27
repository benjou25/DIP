import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, exposure

# Load the input image
image_path = './pics/xRayChest.tif'
tif_image = io.imread(image_path)

# Convert the image to grayscale if it has multiple channels
if tif_image.ndim == 3:  # Check if the image has multiple channels
    tif_gray = color.rgb2gray(tif_image)
else:
    tif_gray = tif_image

# Calculate the histogram of the input image
hist_original, bins_original = np.histogram(tif_gray, bins=256, range=(0, 255))

# Calculate the cumulative distribution function (CDF)
cdf = np.cumsum(hist_original)

# Normalize the CDF to span the entire intensity range (0 to 255)
cdf_normalized = cdf / cdf[-1]

# Create a lookup table to map pixel values
equalization_lookup = (cdf_normalized * 255).astype(np.uint8)

# Apply the lookup table to the image to perform equalization
equalized_image = equalization_lookup[tif_gray]

# Perform histogram equalization using a built-in function
equalized_image_builtin = exposure.equalize_hist(tif_gray)

# Calculate the histogram of the equalized image
hist_equalized, bins_equalized = np.histogram(equalized_image, bins=256, range=(0, 255))

cdf2 = np.cumsum(hist_equalized)
cdf2_normalized = cdf2 / cdf2[-1]

# Plot the original histogram
plt.figure(figsize=(18, 4))
plt.subplot(131)
plt.hist(tif_gray.ravel(), bins=256, range=(0, 255))
plt.title('Original Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

# Plot the equalized histogram
plt.subplot(132)
plt.hist(equalized_image.ravel(), bins=256, range=(0, 255))
plt.title('Equalized Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

# Plot the cumulative density function (CDF)
plt.subplot(133)
plt.plot(bins_equalized[:-1], cdf2_normalized, color='blue')
plt.title('Cumulative Density')
plt.xlabel('Pixel Value')
plt.ylabel('CDF')

# Plot the cumulative density function after equalization (CDF)
plt.subplot(133)
plt.plot(bins_original[:-1], cdf_normalized, color='black')
plt.title('Cumulative Density equalized')
plt.xlabel('Pixel Value')
plt.ylabel('CDF')

plt.tight_layout()
plt.show()

#-----------------------------------------------------------------------------------------

# Plot the original, manually equalized, and built-in equalized images
plt.figure(figsize=(18, 4))
plt.subplot(131)
plt.imshow(tif_gray, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(132)
plt.imshow(equalized_image, cmap='gray')
plt.title('Manually Equalized Image')
plt.axis('off')

plt.subplot(133)
plt.imshow(equalized_image_builtin, cmap='gray')
plt.title('Built-in Equalized Image')
plt.axis('off')

plt.tight_layout()
plt.show()