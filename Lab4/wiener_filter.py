from pathlib import Path
import funcs
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage, fft
from PIL import Image

# ----------------------------------------------------- load image ---------------------------------------------
file_name = Path('.', 'pics', 'blurred_image.jpg')

# Open the image with pillow and convert to numpy array
image = Image.open(file_name)
im_color = np.asarray(Image.open(file_name))
im_gray = np.asarray(Image.open(file_name).convert('L'))

# summarize some details about the image
print(image.format)
print('dtype image:', im_gray.dtype)
print(im_gray.shape)

# motion blur filter
nFilter = 91
angle = 45
my_filter = np.zeros((nFilter, nFilter))
my_filter[nFilter//2, :] = 1.0 / nFilter
my_filter = ndimage.rotate(my_filter, angle, reshape=False)

nRows = im_gray.shape[0]
nCols = im_gray.shape[1]
nFFT = 2048

im_spect = fft.fft2(im_gray, (nFFT, nFFT))
mag_fft = np.abs(fft.fftshift(im_spect))
log_mag_fft = np.log(1 + mag_fft)

# -------------------------- wiener filter ---------------------
# Define the Wiener filter parameters
K = 0.01  # A small constant to prevent division by zero
H = np.fft.fft2(my_filter, (nFFT, nFFT))  # Fourier transform of the filter
H_conj = np.conj(H)      # complex conjugate of the filter
# Wiener filter formula
spec_rec = (H_conj / (np.abs(H) ** 2 + K)) * im_spect
# Inverse Fourier transform to get the reconstructed image
im_rec = scipy.fft.ifft2(spec_rec)
im_rec = np.real(im_rec)[:nRows-nFilter, :nCols-nFilter]
# Fourier transform of the reconstructed image
mag_spec_rec = np.abs(fft.fftshift(spec_rec))
log_spec_rec = np.log(1 + mag_spec_rec)

# --------------------------------------------------------- display images ---------------------------------------------
fig0 = plt.figure(1)
plt.subplot(1, 2, 1)
plt.title('blurred Image')
plt.imshow(im_gray, cmap='gray')

plt.subplot(1, 2, 2)
plt.title('spectrum')
plt.imshow(log_mag_fft, cmap='gray')

plt.tight_layout()
for ax in fig0.get_axes():
    ax.axis('off')
plt.show()
plt.show()

# ------------------------------------ reconstructed image -------------------------------------
fig1 = plt.figure(2)
plt.subplot(1, 2, 1)
plt.title('reconstructed image')
plt.imshow(im_rec, cmap='gray')

plt.subplot(1, 2, 2)
plt.title('reconstructed spectrum')
plt.imshow(log_spec_rec, cmap='gray')

plt.tight_layout()
for ax in fig1.get_axes():
    ax.axis('off')
plt.show()
plt.show()