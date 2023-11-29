import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# Check OpenCV version
cv_version = cv2.__version__.split('.')[0]
if cv_version == '3':
    sift = cv2.xfeatures2d.SIFT_create()
elif cv_version == '4':
    sift = cv2.SIFT_create()

# Load the reference images
staple_image = cv2.imread('stapleRemover.jpg', 0)
desk_image = cv2.imread('clutteredDesk.jpg', 0)

# Find keypoints and descriptors with SIFT for the reference images
kp1, des1 = sift.detectAndCompute(staple_image, None)
kp2, des2 = sift.detectAndCompute(desk_image, None)

# Brute Force matcher
start_time_bf = time.time()
bf = cv2.BFMatcher()
matches_bf = bf.knnMatch(des1, des2, k=2)
good_matches_bf = [m for m, n in matches_bf if m.distance < 0.75 * n.distance]
end_time_bf = time.time()
computation_time_bf = end_time_bf - start_time_bf

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

# Initialize FLANN matcher
start_time_flann = time.time()
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches_flann = flann.knnMatch(des1, des2, k=2)
good_matches_flann = [m for m, n in matches_flann if m.distance < 0.75 * n.distance]
end_time_flann = time.time()
computation_time_flann = end_time_flann - start_time_flann

# Draw matches for Brute Force
img_matches_bf = cv2.drawMatches(staple_image, kp1, desk_image, kp2, good_matches_bf, None,
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Draw matches for FLANN
img_matches_flann = cv2.drawMatches(staple_image, kp1, desk_image, kp2, good_matches_flann, None,
                                     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Convert images to RGB for displaying with plt.imshow
img_matches_bf_rgb = cv2.cvtColor(img_matches_bf, cv2.COLOR_BGR2RGB)
img_matches_flann_rgb = cv2.cvtColor(img_matches_flann, cv2.COLOR_BGR2RGB)

# Display both sets of matches with computation time in titles
plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)
plt.imshow(img_matches_bf_rgb)
plt.title(f'Matches with Brute Force Matcher\nComputation Time: {computation_time_bf:.4f} seconds')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_matches_flann_rgb)
plt.title(f'Matches with FLANN Matcher\nComputation Time: {computation_time_flann:.4f} seconds')
plt.axis('off')

plt.show()