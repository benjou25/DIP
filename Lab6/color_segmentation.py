from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from PIL import Image

# Load the images
file_name = Path('.', 'pics', 'brainCells.tif')
cells = Image.open(file_name)

# extract RGB-channels
rgb_cells = cells.convert('RGB')
R = np.array(rgb_cells)[:,:,0]
G = np.array(rgb_cells)[:,:,1]
B = np.array(rgb_cells)[:,:,2]

# Create the CMY channels
C = 255 - R
M = 255 - G
Y = 255 - B

# ----------------------------------- plots -------------------------------------------------
fig1, axes1 = plt.subplots(2, 3, figsize=(12, 8))
# Plot the RGB-cchannels
axes1[0,0].imshow(R, cmap='gray')
axes1[0,0].set_title('R')
axes1[0,0].axis('off')
axes1[0,1].imshow(G, cmap='gray')
axes1[0,1].set_title('G')
axes1[0,1].axis('off')
axes1[0,2].imshow(B, cmap='gray')
axes1[0,2].set_title('B')
axes1[0,2].axis('off')
# Plot the CMY-cchannels
axes1[1,0].imshow(C, cmap='gray')
axes1[1,0].set_title('C')
axes1[1,0].axis('off')
axes1[1,1].imshow(M, cmap='gray')
axes1[1,1].set_title('M')
axes1[1,1].axis('off')
axes1[1,2].imshow(Y, cmap='gray')
axes1[1,2].set_title('Y')
axes1[1,2].axis('off')
plt.show()