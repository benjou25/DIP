

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.image as img
import cv2
import skimage        #<---
from skimage import data

import pydicom as dicom
import os
from PIL import Image

from pathlib import Path



#DataType__ImageName_ImageModifications




#Task 1.1
#----------------------------------------------------------------------------------------------------
#Load and display Image
#Generate grayscale version of the Image
#Display the shape of the Image


image__manInDesert = img.imread("./pics/MenInDesert.jpg")
image__manInDesert__grayscale = np.asarray(Image.open("./pics/MenInDesert.jpg").convert('L'))

print("")
print("Task 1.1:")
print("Original Image: Shape: ", image__manInDesert.shape)

plt.figure(11) 
plot = plt.subplot(1,1,1)
plt.title("Orignal Image")
plt.imshow(image__manInDesert, cmap='gray')


#plt.show()
#----------------------------------------------------------------------------------------------------









#Task 1.2
#----------------------------------------------------------------------------------------------------
#Compute FFT2 of the Image (F)
#Print the data type of the FFT2 Image


#Transformed Image (F)
fft__manInDesert__grayscale = sp.fft.fft2(image__manInDesert__grayscale, (image__manInDesert__grayscale.shape[0],image__manInDesert__grayscale.shape[1]))

#plt.figure(12) 
#plot = plt.subplot(1,1,1)
#plt.title("Orignal Image_FFT")
#plt.imshow(image_manInDesert_fft, cmap='gray')

print("")
print("Task 1.2:")
print('Original Image FFT: Data Type: ', fft__manInDesert__grayscale.dtype)



#plt.show()
#----------------------------------------------------------------------------------------------------









#Task 1.3
#----------------------------------------------------------------------------------------------------
#Display Original Image and Magitude of the FFT2 of the Image (in linear and log) 


fft_magnitude_lin__manInDesert__grayscale = np.abs(sp.fft.fftshift(fft__manInDesert__grayscale))
fft_magnitude_log__manInDesert__grayscale = np.log(1+fft_magnitude_lin__manInDesert__grayscale)

plt.figure(13) 

plot = plt.subplot(3,1,1)
plt.title("Original Image")
plt.imshow(image__manInDesert, cmap='gray')
plt.axis("off")

plot = plt.subplot(3,1,2)
plt.title("Magnitude of Fourier transform (Linear)")
plt.imshow(fft_magnitude_lin__manInDesert__grayscale, cmap='gray')
plt.axis("off")

plot = plt.subplot(3,1,3)
plt.title("Magnitude of Fourier transform (Logarithmic)")
plt.imshow(fft_magnitude_log__manInDesert__grayscale, cmap='gray')
plt.axis("off")
plt.tight_layout()

#plt.show()
#----------------------------------------------------------------------------------------------------









#Task 1.4
#----------------------------------------------------------------------------------------------------
#Transform the FFT2 back to real space with ifft2()
#Print data Type of the Original Image and the ifft2 Image


fft_realSpace__manInDesert__grayscale = sp.fft.ifft(fft__manInDesert__grayscale)

print("")
print("Task 1.4:")
print('Backtransformed FFT Image: Data Type: ', fft_realSpace__manInDesert__grayscale.dtype)
print('Original Image FFT: Data Type: ', fft__manInDesert__grayscale.dtype)
print("How can the original data type be obtained? .............")

#print('Modified_Image_Spectrum: Data Type: ', modified_image_spectrum.dtype)
#----------------------------------------------------------------------------------------------------









#Task 1.5
#----------------------------------------------------------------------------------------------------
#Create a 9x9 averaging kernal and apply it to the Original Image


# Averaging Filter
def create_normalized_averaging_kernel(size):
    #This function creates a n sized kernel for a averaging filter
    kernal = np.ones((size,size))
    kernal_normalized = (1/sum(kernal))*kernal
    return kernal_normalized

kernel_size = 9
averaging_kernel_normalized = create_normalized_averaging_kernel(kernel_size)

image__manInDesert__grayscale_averaging = sp.signal.convolve2d(image__manInDesert__grayscale, averaging_kernel_normalized)

fig = plt.figure(15)

plt.subplot(1, 1, 1)
plt.title('Averaging Filter 9x9 over Original Image')
plt.imshow(image__manInDesert__grayscale_averaging, cmap='gray')
plt.axis("off")

#plt.show()
#----------------------------------------------------------------------------------------------------









#Task 1.6
#----------------------------------------------------------------------------------------------------
#Compute the spectrum (H) of of the Filtered Image
#Multiply F with H to get G


#Filtered Spectrum (H)
fft__manInDesert__grayscale_averaging = np.fft.fft2(image__manInDesert__grayscale_averaging, (image__manInDesert__grayscale.shape[0],image__manInDesert__grayscale.shape[1]))

#Transformed Image (F)
#fft__manInDesert__grayscale      see Task 1.2
#fft_zeroPadded__manInDesert__grayscale = np.pad(fft__manInDesert__grayscale, (4,4))


#.....(G)
fft__manInDesert__grayscale_averaging_multiplied_fft_Original = fft__manInDesert__grayscale*fft__manInDesert__grayscale_averaging


#Coversion to real space
ifft__manInDesert__grayscale_averaging_multiplied_fft_Original = sp.fft.ifft2(fft__manInDesert__grayscale_averaging_multiplied_fft_Original)

nRows__manInDesert = image__manInDesert__grayscale.shape[0]
nCols__manInDesert = image__manInDesert__grayscale.shape[1]
image__manInDesert__grayscale_averaging_multiplied_fft_Original = np.real(ifft__manInDesert__grayscale_averaging_multiplied_fft_Original)[kernel_size:nRows__manInDesert+kernel_size, kernel_size:nCols__manInDesert+kernel_size]

#modified_image_2 = np.real(image_grayPixels_averaging_2)[k_size:nRows+k_size, k_size:nCols+k_size]

fig = plt.figure(16)

plt.subplot(1, 1, 1)
plt.title('Image G via frequency space')
plt.imshow(image__manInDesert__grayscale_averaging_multiplied_fft_Original, cmap='gray')
plt.axis("off")

#plt.show()



#?????????? Plot falsch?
#----------------------------------------------------------------------------------------------------









#Task 1.7
#----------------------------------------------------------------------------------------------------

#???????????? 

#----------------------------------------------------------------------------------------------------









#Task 1.8
#----------------------------------------------------------------------------------------------------

#???????????? 

#----------------------------------------------------------------------------------------------------






























#Task 2
#----------------------------------------------------------------------------------------------------
# ----------------------------------------------------- load image ---------------------------------------------
file_name = Path('.', 'pics', 'MenInDesert.jpg')

# Open the image with pillow and convert to numpy array
image = Image.open(file_name)
color_pixels = np.asarray(Image.open(file_name))
gray_pixels = np.asarray(Image.open(file_name).convert('L'))

# summarize some details about the image
print("")
print("Task 2:")
print(image.format)
print('numpy array:', gray_pixels.dtype)
print(gray_pixels.shape)

# -------------------------------------------- generate the motion blur filter -----------------------------------------
nFilter = 91
angle = 30
my_filter = np.zeros((nFilter, nFilter))
my_filter[nFilter//2, :] = 1.0 / nFilter
my_filter = sp.ndimage.rotate(my_filter, angle, reshape=False)

nRows = gray_pixels.shape[0]
nCols = gray_pixels.shape[1]
nFFT = 1024

image_spectrum = sp.fft.fft2(gray_pixels, (nFFT, nFFT))
filter_spectrum = sp.fft.fft2(my_filter, (nFFT, nFFT))

print('Image Spectrum: Data Type: ', image_spectrum.dtype)

modified_image_spectrum = image_spectrum * filter_spectrum
modified_image = sp.fft.ifft2(modified_image_spectrum)
modified_image = np.real(modified_image)[nFilter:nRows + nFilter, nFilter:nCols + nFilter]


# --------------------------------------------------- reconstruct the image --------------------------------------------
# here goes your code ...

K = 1
filter_size = 91
filter_shiftAngle = 45
filter_shiftDistance = 10






image__blurred_image = img.imread("./pics/blurred_image.jpg")
image__blurred_image__grayscale = np.asarray(Image.open("./pics/blurred_image.jpg").convert('L'))


nFFT_x2 = image__blurred_image.shape[0]
nFFT_y2 = image__blurred_image.shape[1]


#g(x,y) is the blurred Image in real space
g = image__blurred_image__grayscale


#G(u,v) is the blurred image in the frequency-domain
#Calculate G(u,v) with the FFT2 (spectrum) of the blurred_image 
G = sp.fft.fft2(g, (nFFT_x2, nFFT_y2))


#h(x,y) is the Motion Blur Filter in the time-domain
reconstruction_filter_0deg = np.zeros((filter_size, filter_size))
reconstruction_filter_0deg[filter_size//2][(filter_size-filter_shiftDistance)//2:(filter_size+filter_shiftDistance)//2] = 1.0 / filter_size
reconstruction_filter_rotated = sp.ndimage.rotate(reconstruction_filter_0deg, filter_shiftAngle, reshape=False)
h = reconstruction_filter_rotated


#H(u,v) is the Motion Blur Filter in the frequency-domain
#Calculate H(u,v) with the FFT2 (spectrum) of the Motion Blur Filter 
H = sp.fft.fft2(h, (nFFT_x2, nFFT_y2))
H__complexConjugate = np.conjugate(H)
H__abs_squared = np.square(abs(H))


#F(u,v) is the reconstructed Image in the frequency-domain
F = (H__complexConjugate*G)/(H__abs_squared+K)



#??????????????????????????????????? Bild wird verschoben nicht verzerrt^^??


#f(x,v) is the reconstructed Image in real space
ifft2__F = sp.fft.ifft2(F)

nRows__F = F.shape[0]
nCols__F = F.shape[1]
f = np.real(ifft2__F)[filter_size:nRows__F+filter_size, filter_size:nCols__F+filter_size]





# --------------------------------------------------------- display images ---------------------------------------------

fig = plt.figure(20)
plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(gray_pixels, cmap='gray')
plt.axis("off")

plt.subplot(2, 2, 2)
plt.title('Motion Blur Filter')
plt.imshow(my_filter, cmap='gray')
plt.axis("off")

plt.subplot(2, 2, 3)
plt.title('Modified Image')
plt.imshow(modified_image, cmap='gray')
plt.axis("off")

plt.tight_layout()



#-------


fig = plt.figure(22)
plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(image__blurred_image__grayscale, cmap='gray')
plt.axis("off")

plt.subplot(2, 2, 2)
plt.title('Reconstruction Filter 0°')
plt.imshow(reconstruction_filter_0deg, cmap='gray')
plt.axis("off")

plt.subplot(2, 2, 4)
plt.title('Reconstruction Filter Rotated')
plt.imshow(h, cmap='gray')
plt.axis("off")


plt.subplot(2, 2, 3)
plt.title('Reconstructed Image')
plt.imshow(f, cmap='gray')
plt.axis("off")

plt.tight_layout()


plt.show()


