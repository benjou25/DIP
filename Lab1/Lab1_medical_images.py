import numpy as np
import matplotlib.pyplot as plt
import pydicom

b1 = pydicom.dcmread('./pics/brain/brain_001.dcm')

# Access metadata elements
patient_name = b1.PatientName
study_date = b1.StudyDate
study_description = b1.StudyDescription
modality = b1.Modality

# Access and manipulate the pixel data
pixel_data = b1.pixel_array
flipped_horizontal = np.fliplr(pixel_data)
flipped_vertical = np.flipud(pixel_data)

# uint8 conversion
pixel_data_uint8 = pixel_data.astype(np.uint8)
pixel_data_float = pixel_data.astype(np.float_)
# normalize before uint8 conversion
min_val = np.min(pixel_data)
max_val = np.max(pixel_data)
pixel_data_normalized = ((pixel_data - min_val) / (max_val - min_val) * 255).astype(np.uint8)

# Create subplots for the original and flipped images
fig1, axes1 = plt.subplots(1, 3, figsize=(10, 10))  # Create a subplot grid

# Display the original image
axes1[0].imshow(pixel_data, cmap='gray')
axes1[0].set_title("Original Image")
axes1[0].axis('off')

# Display the horizontally flipped image
axes1[1].imshow(flipped_horizontal, cmap='gray')
axes1[1].set_title("Flipped Horizontal")
axes1[1].axis('off')

# Display the vertically flipped image
axes1[2].imshow(flipped_vertical, cmap='gray')
axes1[2].set_title("Flipped Vertical")
axes1[2].axis('off')

plt.tight_layout()
plt.show()

# Create subplots for the uint8 converted images
fig2, axes2 = plt.subplots(1, 3, figsize=(10, 10))  # Create a subplot grid

# Display the uint8 converted image
axes2[0].imshow(pixel_data_uint8, cmap='gray')
axes2[0].set_title("Image (uint8) not normalized")
axes2[0].axis('off')

# Display the uint8 normalized image
axes2[1].imshow(pixel_data_normalized, cmap='gray')
axes2[1].set_title("Image (uint8) normalized")
axes2[1].axis('off')

# Display the uint8 normalized image
axes2[2].imshow(pixel_data_float, cmap='gray')
axes2[2].set_title("Image (float)")
axes2[2].axis('off')

plt.tight_layout()
plt.show()

# Print metadata
print("Patient Name:", patient_name)
print("Study Date:", study_date)
print("Study Description:", study_description)
print("Modality:", modality)