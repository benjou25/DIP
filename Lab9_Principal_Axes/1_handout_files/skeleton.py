import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import cv2
from numpy.linalg import eig


##### Aspect Ratio and Extent #####

# 1. Read image
img_bone = plt.imread('bone.tif')
img_bone_gray = cv2.cvtColor(img_bone, cv2.COLOR_RGBA2GRAY)
img_bone_gray = img_bone_gray.astype(np.float64)/np.max(img_bone_gray)
plt.figure()
plt.imshow(img_bone_gray, cmap='gray')
plt.show()


# 2. Create a matrix A with the coordinates of the pixels that belong to the bone and whose centroid is in the origin
points = np.argwhere(img_bone_gray > 0.5)
# Note that the matrix points have the dimension (number of image points x 2)
#    points[:,0] ... y coordinates
#    points[:,1] ... x coordinates
# Change the order so that 
#    points[:,0] ... x coordinates
#    points[:,1] ... y coordinates
points = points[:, ::-1]

center_x = np.mean(points[:,0])
center_y = np.mean(points[:,1])
x_0 = int(np.round(center_x))
y_0 = int(np.round(center_y))
center = [x_0 , y_0]

points = points - center

covarianceMatrix = np.cov(points, rowvar=False)


# 3. Compute the eigen vector for the largest eigen value
D, V = eig(covarianceMatrix)
idx = np.argsort(D)[-1]  # Index of largest eigenvalue
eigV = V[:, idx]  # Eigen vector for largest eigen value

# 4. Compute rotation that aligns the bone's dominant axis to the image's x-axis
height, width = img_bone_gray.shape
# 4.a) Method using the explicit rotation angle phi
offset_x, offset_y = -50, 200
rotation_angle = -np.arctan2(eigV[1], eigV[0])  # Calculate the rotation angle
translation_offset = np.array([offset_x, offset_y])  # Replace offset_x and offset_y with your desired values
rotMatAffine = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle), translation_offset[0]],
                        [np.sin(rotation_angle), np.cos(rotation_angle), translation_offset[1]]])

# 5. Rotate the image
img_bone_rot = cv2.warpAffine(img_bone_gray, rotMatAffine, (width, height), cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
plt.figure()
plt.imshow(img_bone_rot, cmap='gray')
plt.title('Rotated Object explicit')
plt.show()

# 6. Determine the bounding box for the aligned bone
alignedBoneCoords = np.argwhere(img_bone_rot > 0.5)
y1 = np.min(alignedBoneCoords[:, 0])
x1 = np.min(alignedBoneCoords[:, 1])
y2 = np.max(alignedBoneCoords[:, 0])
x2 = np.max(alignedBoneCoords[:, 1])
# 7. Draw the bonding box onto the image
img_bone_rot = cv2.rectangle(img_bone_rot, (x1, y1), (x2, y2), color=1)

height = y2-y1+1
width = x2-x1+1
print('Bounding Box: width={}, height={}'.format(width, height))

# 8. Plot the aligned object bone including bounding box
plt.figure()
plt.imshow(img_bone_rot, cmap='gray')
plt.title('Rotated Object with Boundig Box')
plt.show()

foo = 0

###### Texture and Co-Ocurrence matrix #####
img_mc1=plt.imread('mc1.tif')
img_mc2=plt.imread('mc2.tif')
img_mc3=plt.imread('mc3.tif')
img_mc4=plt.imread('mc4.tif')

img_ut = img_mc1.copy()
height, width = img_ut.shape

resizeParam=1
#img_ut = cv2.resize(img_ut, (resizeParam*width, resizeParam*height))

plt.figure()
plt.imshow(img_ut, cmap='gray')
plt.show()

# Computation of the Co-Ocurrence Matrix
cooMat=np.zeros((256,256))

M,N = img_ut.shape

count=1
for ii in range(1,M-1):
    for jj in range(1,N-1):
        # Define the pixel values at the current and neighboring locations
        p = img_ut[ii, jj]

        q = img_ut[ii, jj + 1]  # Right neighbor
        cooMat[p, q] += 1

        q = img_ut[ii + 1, jj]  # Bottom neighbor
        cooMat[p, q] += 1

        q = img_ut[ii, jj - 1]  # Left neighbor
        cooMat[p, q] += 1

        q = img_ut[ii - 1, jj]  # Upper neighbor
        cooMat[p, q] += 1
        count = count + 4 
cooMat = cooMat/count


# Plotting the Co-Ocurrence matrix 
X = np.arange(cooMat.shape[1])
Y = np.arange(cooMat.shape[0])
X, Y = np.meshgrid(X, Y)
fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, cooMat, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()


energy = 0
contrast = 0
entropy = 0
homogenity = 0

for ii in range(256):
    for jj in range(256):
        energy += (cooMat[ii, jj])**2
        contrast += ((ii-jj)**2) * cooMat[ii,jj]
        # Avoid taking the logarithm of zero
        if cooMat[ii, jj] > 0:
            entropy += cooMat[ii, jj] * np.log2(cooMat[ii, jj])
        homogenity += cooMat[ii,jj] / (1 + np.abs(ii-jj))

entropy = -entropy        
print("energy: %5.5f\ncontrast: %5.5f\nentropy: %5.5f\nhomogenity: %5.5f"%(energy,contrast,entropy,homogenity))
