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

value_counts = {}

# Loop through and display all TIFF images
for i, file_path in enumerate(tif_files):
    try:
        # Open the TIFF file
        tif_image = io.imread(file_path)
        
        # Convert the TIFF image to grayscale (if not already)
        if tif_image.ndim == 3:  # Check if the image has multiple channels
            tif_image_gray = color.rgb2gray(tif_image)
        else:
            tif_image_gray = tif_image
        
        # Append the grayscale TIFF image to the images array
        images.append(tif_image_gray)  
        
        # Create a subplot for each image
        plt.subplot(1, 3, i + 1)  # 1 row, 3 columns for a 1x3 grid
        plt.imshow(tif_image_gray, cmap='gray')
        plt.title(f"Image {i + 1}")
        plt.axis('off')
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
plt.tight_layout()
plt.show()

for i in range(np.size(images)):
    # Flatten the 2D grayscale image into a 1D array
    pixel_values = images[0].flatten()
            
    # Count the occurrences of each pixel value and update the dictionary
    for pixel_value in pixel_values:
        if pixel_value in value_counts:
            value_counts[pixel_value] += 1
        else:
            value_counts[pixel_value] = 1

    values = np.array(list(value_counts.keys()))
    counts = np.array(list(value_counts.values()))

    plt.subplot(1, 3, i + 1)  # 1 row, 3 columns for a 1x3 grid
    plt.bar(values, counts)
    plt.title(f"Grayscale Value Counts Histogram {i + 1}")
    plt.axis('off')
    plt.show()

# Create subplots
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