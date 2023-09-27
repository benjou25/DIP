import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
import os

# import camel
camel = io.imread(".\Bilder\camel.jpg")
camel_gray = color.rgb2gray(camel)

# Directory containing your TIFF images
image_directory = './pics/'

# List all TIFF files in the directory and create an array
tif_files = [os.path.join(image_directory, filename) for filename in os.listdir(image_directory) if filename.endswith('.tif')]
images = []
hist_val = []
hist_cnt = []

# Create a figure for the subplots
fig, axes = plt.subplots(len(tif_files), 3, figsize=(12, 6 * len(tif_files)))

# Loop through and display all TIFF images
for i, file_path in enumerate(tif_files):
    try:
        # Open the TIFF file
        tif_image = io.imread(file_path)
        value_counts = {}
        
        # Convert the TIFF image to grayscale (if not already)
        if tif_image.ndim == 3:  # Check if the image has multiple channels
            tif_image_gray = color.rgb2gray(tif_image)
        else:
            tif_image_gray = tif_image

        pixel_values = tif_image_gray.flatten()

        # count the different grayscale values
        for pixel_value in pixel_values:
            if pixel_value in value_counts:
                value_counts[pixel_value] += 1
            else:
                value_counts[pixel_value] = 1

        # get the values and their frequency
        values = np.array(list(value_counts.keys()))
        counts = np.array(list(value_counts.values()))
        
        # Append the grayscale TIFF image to the images array
        images.append(tif_image_gray)
        # Append values and counts of the image to corresponding array
        hist_val.append(values)
        hist_cnt.append(counts)
        
        # Create a subplot for each image
        axes[i, 0].imshow(tif_image_gray, cmap='gray')
        axes[i, 0].set_title(f"Image {i + 1}")
        axes[i, 0].axis('off')
        
        # Create a histogram subplot for the image
        axes[i, 1].bar(hist_val[i], hist_cnt[i])
        axes[i, 1].set_title(f"Histogram {i + 1}")

        # Create histogram with function
        axes[i, 2].hist(tif_image_gray.ravel(), bins=256, color='gray', alpha=0.7)
        axes[i, 2].set_title(f"Histogram {i + 1}")
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

# Add a title to the figure
fig.suptitle("Grayscale Images and Histograms", fontsize=16)
# Adjust layout for better readability
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

# Create subplots for camel image
fig1, axes1 = plt.subplots(1, 2, figsize=(10, 6))  # Create a subplot grid
# Display greyscale
axes1[0].imshow(camel_gray, cmap='gray')
axes1[0].set_title("Es Kamel")
axes1[0].axis('off')
# Display color
axes1[1].imshow(camel, cmap='gray')
axes1[1].set_title("Es Kamel")
axes1[1].axis('off')

plt.show()