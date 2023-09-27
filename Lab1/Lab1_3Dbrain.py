import numpy as np
import webbrowser
import matplotlib.pyplot as plt
from PIL import Image
import pydicom
import os
import func

# Directory containing your DICOM images
image_directory = './pics/brain/'

# List all DICOM files in the directory and create array
dicom_files = [os.path.join(image_directory, filename) for filename in os.listdir(image_directory) if filename.endswith('.dcm')]
images = []

# Loop through and display all DICOM images
for i, file_path in enumerate(dicom_files):
    # Load DICOM file
    dicom_data = pydicom.dcmread(file_path)
    
    # Access and normalize pixel data
    pixel_data = dicom_data.pixel_array
    min_val = np.min(pixel_data)
    max_val = np.max(pixel_data)
    pixel_data_normalized = ((pixel_data - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    # Append the normalized pixel data to the images array
    images.append(pixel_data_normalized)

    # Create a subplot for each image
    plt.subplot(4, 5, i + 1)  # 4 rows, 5 columns for a 4x5 grid
    plt.imshow(pixel_data_normalized, cmap='gray')
    plt.title(f"Image {i + 1}")
    plt.axis('off')

# Convert the images list to a 3D NumPy array
images_array = np.array(images)

# save all images in a .gif file
sagittal_obj = [Image.fromarray(image, mode='L') for image in images_array]
sagittal_obj[0].save('brain.gif', save_all=True, append_images=sagittal_obj[1:], duration=100, loop=0)

# Adjust layout and show the subplots
plt.tight_layout()
plt.show()

# Now, you can access individual images using the third index of the images_array.
# For example, to access the first image, use images_array[0,:,:].

pix = np.size(images_array[0,:,0])
slicex = np.zeros((20, pix))
slicey = np.zeros((20, pix))

sagittal = []
frontal  = []
for j in range(32):
    for n in range(np.size(images_array[:,0,0])):
        slicex[n, :] = images_array[n,:,8*j]     # sagittal
        slicey[n, :] = images_array[n,8*j,:]     # frontal

    # stretching the image to double size 3 times -> 2^3 x stretched
    rescalex, rescaley = func.stretch(3,slicex,slicey)

    sagittal.append(np.flipud(rescalex))
    frontal.append(np.flipud(rescaley))

sagittal_array = np.array(sagittal)
frontal_array = np.array(frontal)
sagittal_images = [Image.fromarray(image.astype('uint8'), mode='L') for image in sagittal_array]
frontal_images = [Image.fromarray(image.astype('uint8'), mode='L') for image in frontal_array]
sagittal_images[0].save('sagittal.gif', save_all=True, append_images=sagittal_images[1:], duration=100, loop=0)
frontal_images[0].save('frontal.gif', save_all=True, append_images=frontal_images[1:], duration=100, loop=0)

# Create subplots for the original and flipped images
fig1, axes1 = plt.subplots(1, 2, figsize=(10, 6))  # Create a subplot grid

# Display sgittal plane rescaled
axes1[0].imshow(sagittal[16], cmap='gray')
axes1[0].set_title("x8 rescaled sagittal plane")
axes1[0].axis('off')

# Display frontal plane rescaled
axes1[1].imshow(frontal[16], cmap='gray')
axes1[1].set_title("x8 rescaled frontal plane")
axes1[1].axis('off')

plt.tight_layout()
plt.show()

webbrowser.open('brain.gif')
webbrowser.open('sagittal.gif')
webbrowser.open('frontal.gif')