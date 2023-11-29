import cv2
import numpy as np

# Load the reference image
reference_image = cv2.imread('stabilo.jpg', 0)

# Initialize the ORB detector
orb = cv2.ORB_create()

# Find keypoints and descriptors with ORB for the reference image
kp1, des1 = orb.detectAndCompute(reference_image, None)

# FLANN parameters
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
search_params = dict(checks=50)

# Initialize FLANN matcher
flann = cv2.FlannBasedMatcher(index_params, search_params)

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened():  # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

# Counter for consecutive frames with object detection
consecutive_frames_with_detection = 0
# Threshold for consecutive detections
consecutive_frames_threshold = 10

while rval:
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find keypoints and descriptors with ORB in the current frame
    kp2, des2 = orb.detectAndCompute(gray, None)

    # Use FLANN matcher to find the best matches between descriptors
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply ratio test to filter good matches
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2 and match_pair[0].distance < 0.7 * match_pair[1].distance:
            good_matches.append(match_pair[0])

    # Draw circles around the matched keypoints in both images
    frame_matches = cv2.drawMatches(reference_image, kp1, gray, kp2, good_matches[:10], None,
                                     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Check if enough good matches are found
    if len(good_matches) > 5:
        consecutive_frames_with_detection += 1

        # Display text when object is consecutively detected
        if consecutive_frames_with_detection >= consecutive_frames_threshold:
            cv2.putText(frame_matches, 'Object Detected!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                        cv2.LINE_AA)

    else:
        consecutive_frames_with_detection = 0

    # Display the frame with feature matches
    cv2.imshow("preview", frame_matches)

    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break

vc.release()
cv2.destroyAllWindows()