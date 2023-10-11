from pathlib import Path
import funcs
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage, fft
from PIL import Image


# ----------------------------------------------------- load image ---------------------------------------------
file_name = Path('.', 'pics', 'MenInDesert.jpg')

# Open the image with pillow and convert to numpy array
image = Image.open(file_name)
color_pixels = np.asarray(Image.open(file_name))
gray_pixels = np.asarray(Image.open(file_name).convert('L'))

# summarize some details about the image
print(image.format)
print('dtype image:', gray_pixels.dtype)
print(gray_pixels.shape)

# -------------------------------------------- generate the motion blur filter -----------------------------------------
nFilter = 91
angle = 30
my_filter = np.zeros((nFilter, nFilter))
my_filter[nFilter//2, :] = 1.0 / nFilter
my_filter = ndimage.rotate(my_filter, angle, reshape=False)

nRows = gray_pixels.shape[0]
nCols = gray_pixels.shape[1]
nFFT = 1024

image_spectrum = fft.fft2(gray_pixels, (nFFT, nFFT))
filter_spectrum = fft.fft2(my_filter, (nFFT, nFFT))

# dtype of the fft2
print('dtype fft2:', image_spectrum.dtype)
# magnitude of the fft
mag_fft = np.abs(fft.fftshift(image_spectrum))
log_mag_fft = np.log(1 + mag_fft)

modified_image_spectrum = image_spectrum * filter_spectrum
modified_image = scipy.fft.ifft2(modified_image_spectrum)
print('dtype transformed back:', modified_image.dtype)
modified_image = np.real(modified_image)[nFilter:nRows + nFilter, nFilter:nCols + nFilter]

# --------------------------------------------------- reconstruct the image --------------------------------------------

#-----1.5-1.6-------
k_size = 9
avg_image, avg_kernel = funcs.my_average_filter(gray_pixels, k_size)

avg_spectrum = fft.fft2(avg_kernel, (nFFT, nFFT))
avg_filtered_im_spec = image_spectrum * avg_spectrum
avg_image_2 = scipy.fft.ifft2(avg_filtered_im_spec)
modified_image_2 = np.real(avg_image_2)[k_size:nRows + k_size, k_size:nCols + k_size]

# --------------------------------------------------------- display images ---------------------------------------------
fig0 = plt.figure(1)
plt.subplot(2, 2, 1)
plt.title('blurred Image')
plt.imshow(avg_image, cmap='gray')

plt.subplot(2, 2, 2)
plt.title('blurred in frequency domain')
plt.imshow(modified_image_2, cmap='gray')

plt.tight_layout()
for ax in fig0.get_axes():
    ax.axis('off')
plt.show()

fig1 = plt.figure(1)
plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(gray_pixels, cmap='gray')

plt.subplot(2, 2, 2)
plt.title('FFT2')
plt.imshow(mag_fft, cmap='gray')

plt.subplot(2, 2, 3)
plt.title('FFT2 log')
plt.imshow(log_mag_fft, cmap='gray')

plt.tight_layout()
for ax in fig1.get_axes():
    ax.axis('off')
plt.show()

fig2 = plt.figure(2)
plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(gray_pixels, cmap='gray')

plt.subplot(2, 2, 2)
plt.title('Motion Blur Filter')
plt.imshow(my_filter, cmap='gray')

plt.subplot(2, 2, 3)
plt.title('Modified Image')
plt.imshow(modified_image, cmap='gray')

plt.subplot(2, 2, 4)
plt.title('Reconstructed Image')
# here goes your reconstructed image

plt.tight_layout()
for ax in fig2.get_axes():
    ax.axis('off')
plt.show()