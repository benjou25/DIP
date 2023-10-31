import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# Load the original image
file_name = Path('.', 'pics', 'landscape_1.png')
original_image = Image.open(file_name)

# Define the gamma value
gamma = 0.6

# Convert the original image to a NumPy array
orig_im_array = np.array(original_image)

# Apply gamma correction to each channel
adjusted_image = np.zeros_like(orig_im_array)
equalized_image = np.zeros_like(orig_im_array)

for channel in range(3):
    # gamma correction for each channel
    adjusted_image[:, :, channel] = np.power(orig_im_array[:, :, channel] / 255.0, gamma) * 255.0
    # equalization for each channel
    equalized_image[:, :, channel] = cv2.equalizeHist(orig_im_array[:, :, channel])

# Apply histogram equalization in the Value-channel of the HSV color space
hsv_image = cv2.cvtColor(orig_im_array, cv2.COLOR_RGB2HSV)
hsv_image[:, :, 2] = cv2.equalizeHist(hsv_image[:, :, 2])
equalized_hsv = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

# Add gamma correction only to a single color channel
adjusted_r = np.zeros_like(orig_im_array)
adjusted_r[:, :, 0] = np.power(orig_im_array[:, :, 0] / 255.0, gamma) * 255.0
adjusted_g = np.zeros_like(orig_im_array)
adjusted_g[:, :, 1] = np.power(orig_im_array[:, :, 1] / 255.0, gamma) * 255.0
adjusted_b = np.zeros_like(orig_im_array)
adjusted_b[:, :, 2] = np.power(orig_im_array[:, :, 2] / 255.0, gamma) * 255.0

# --------------------------------------- plots ----------------------------------------------------

# Create a figure with two subplots to compare the images
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
# Plot the original image
axes[0].imshow(original_image)
axes[0].set_title('Original Image')
axes[0].axis('off')
# Plot the adjusted image
axes[1].imshow(adjusted_image)
axes[1].set_title('Gamma Adjusted Image')
axes[1].axis('off')
plt.show()

# Create a figure with two subplots to compare the images
fig2, axes2 = plt.subplots(1, 3, figsize=(12, 6))
# Plot the adjusted image with only the red channel corrected
axes2[0].imshow(adjusted_r)
axes2[0].set_title('Gamma Adjusted Red')
axes2[0].axis('off')
# Plot the adjusted image with only the red channel corrected
axes2[1].imshow(adjusted_g)
axes2[1].set_title('Gamma Adjusted Green')
axes2[1].axis('off')
# Plot the adjusted image with only the red channel corrected
axes2[2].imshow(adjusted_b)
axes2[2].set_title('Gamma Adjusted Blue')
axes2[2].axis('off')
plt.show()

# Create a figure with two subplots to compare the images
fig3, axes3 = plt.subplots(1, 3, figsize=(12, 6))
# Plot the original image
axes3[0].imshow(original_image)
axes3[0].set_title('Original Image')
axes3[0].axis('off')
# Plot the equalized image
axes3[1].imshow(equalized_image)
axes3[1].set_title('Equalized Image (RGB)')
axes3[1].axis('off')
# Plot the equalized image in hsv space
axes3[2].imshow(equalized_hsv)
axes3[2].set_title('Equalized Image (HSV)')
axes3[2].axis('off')
plt.show()