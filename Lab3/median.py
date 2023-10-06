import numpy as np
from skimage import io, color
from scipy.ndimage import median_filter
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import time

# Load the image
imdir = './tifs/cellsSandP.tif'
tif_image = io.imread(imdir)

# Convert the image to grayscale if it's in color
if tif_image.ndim == 3:
    tif_gray = color.rgb2gray(tif_image)
else:
    tif_gray = tif_image

# Define the size of the median filter (e.g., 3x3 or 5x5)
filter_size = 3

# Apply the median filter
start_time = time.time()
median_filtered_image = median_filter(tif_gray, size=filter_size)
end_time = time.time()

# Apply the averaging filter for comparison
kernel = np.ones((filter_size, filter_size), dtype=np.float32) / (filter_size**2)
start_time_linear = time.time()
linear_filtered_image = convolve2d(tif_gray, kernel)
end_time_linear = time.time()

# Display the original and filtered images
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.imshow(tif_gray, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(median_filtered_image, cmap='gray')
plt.title('Median Filtered Image')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(linear_filtered_image, cmap='gray')
plt.title('Averaging Filtered Image')
plt.axis('off')

plt.show()

# Print computation times
median_filter_time = end_time - start_time
linear_filter_time = end_time_linear - start_time_linear
print(f"Median Filter Computation Time: {median_filter_time:.4f} seconds")
print(f"Linear Filter Computation Time: {linear_filter_time:.4f} seconds")