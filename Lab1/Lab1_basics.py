import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io, color

#Import with PIL.Image
im_g1 = Image.open(".\pics\lena_gray.gif")
im_c1 = Image.open(".\pics\lena_color.gif")
#convert to RGB and gray
im_c1_rgb = im_c1.convert("RGB")
im_c1_gray = im_c1.convert("L")

#Import with skimage.io
im_g2 = io.imread('.\pics\lena_gray.gif')
im_c2 = io.imread('.\pics\lena_color.gif')
#convert to RGB and gray
im_c2_rgb = color.rgba2rgb(im_c2)
im_c2_gray = color.rgb2gray(im_c2_rgb)

#Save in different formats
#PIL files:
im_g1.save("./new_pics/gray1.jpg")
im_c1_rgb.save("./new_pics/color1.tiff")
im_c1_gray.save("./new_pics/conv_gray1.png")
#skimage files:
io.imsave("./new_pics/gray2.png", im_g2)
io.imsave("./new_pics/color2.jpg", im_c2_rgb)
io.imsave("./new_pics/conv_gray2.jpg", im_c2_gray)

# Split the PIL image into color channels
r, g, b = im_c1_rgb.split()

print('size color with PIL:',im_c1_rgb.size)
print('size gray with PIL:',im_g1.size)
print('color mode PIL (color)',im_c1_rgb.mode)
print('color mode PIL (gray)',im_g1.mode)
print('\n')
print('size color with skimage:',im_c2_rgb.shape)
print('size gray with skimage:',im_g2.shape)
print('color channels in skimage (color)', im_c2_rgb.shape[2] if len(im_c2_rgb.shape) == 3 else 1)
print('color channels in skimage (gray)', im_g2.shape[2] if len(im_g2.shape) == 3 else 1)

#Image plotting with matplotlib
fig = plt.figure(figsize=(10, 7))
rows = 2
columns = 2

fig.add_subplot(rows, columns, 1)            # subplot von PIL color
plt.imshow(im_c1_rgb)
plt.axis('off')
plt.title("color (PIL)")

fig.add_subplot(rows, columns, 2)            # subplot von PIL gray
plt.imshow(im_g1, cmap='gray')
plt.axis('off')
plt.title("gray (PIL)")

fig.add_subplot(rows, columns, 3)            # subplot von skimage color
plt.imshow(im_c2)
plt.axis('off')
plt.title("color (skimage)")

fig.add_subplot(rows, columns, 4)            # subplot von skimage gray
plt.imshow(im_g2, cmap='gray')
plt.axis('off')
plt.title("gray (skimage)")

plt.show()

#RGB plotting with matplotlib
fig2 = plt.figure(figsize=(10, 7))
rows = 2
columns = 2

fig2.add_subplot(rows, columns, 1)            #PIL color
plt.imshow(im_c1_rgb)
plt.axis('off')
plt.title("Original")

fig2.add_subplot(rows, columns, 2)            # Red
plt.imshow(r, cmap='viridis')
plt.axis('off')
plt.title("Red")

fig2.add_subplot(rows, columns, 3)            # Green
plt.imshow(g, cmap='viridis')
plt.axis('off')
plt.title("Green")

fig2.add_subplot(rows, columns, 4)            # Blue
plt.imshow(b, cmap='viridis')
plt.axis('off')
plt.title("Blue")

plt.show()