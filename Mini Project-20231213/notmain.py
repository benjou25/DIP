import cv2
import numpy as np
import time

def find_strongest_rectangle(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and help contour detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edged image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours to find closed rectangles
    closed_rectangles = []
    for contour in contours:
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Filter rectangles based on the number of vertices and being closed
        if len(approx) == 4 and cv2.isContourConvex(approx):
            closed_rectangles.append(approx)

    # Find the largest closed rectangle
    strongest_rectangle = max(closed_rectangles, key=cv2.contourArea, default=None)

    return strongest_rectangle

# Define the aspect ratio of a standard piece of paper (A4)
paper_aspect_ratio = 1 / 1.414

# Open the webcam (usually 0 or 1, depending on your setup)
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
time.sleep(0.1)
print(cap.isOpened())

# Set a fixed window size
cv2.namedWindow('Webcam with Closed Rectangles', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Webcam with Closed Rectangles', 800, 600)

cv2.namedWindow('Warped Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Warped Image', 800, 600)

# Define colors for each corner (BGR format)
corner_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Find the strongest closed rectangle in the frame
    strongest_rectangle = find_strongest_rectangle(frame)

    # Draw contours on the frame
    frame_contours = frame.copy()
    if strongest_rectangle is not None:
        cv2.drawContours(frame_contours, [strongest_rectangle], -1, (0, 255, 0), 2)

        # Mark corners with different colors
        for i in range(4):
            color = corner_colors[i]
            cv2.circle(frame_contours, tuple(strongest_rectangle[i][0]), 5, color, -1)

        # Apply perspective transformation to the frame
        rect_corners = strongest_rectangle.reshape(-1, 2)
        target_width = max(np.linalg.norm(rect_corners[0] - rect_corners[1]),
                           np.linalg.norm(rect_corners[2] - rect_corners[3]))
        target_height = target_width / paper_aspect_ratio
        target_corners = np.float32([[0, 0], [0, target_height], [target_width, target_height], [target_width, 0]])
        matrix = cv2.getPerspectiveTransform(rect_corners.astype(np.float32), target_corners)
        result = cv2.warpPerspective(frame, matrix, (int(target_width), int(target_height)))
        gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        # Convert the warped image to binary
        _, binary_result = cv2.threshold(gray_result, 128, 255, cv2.THRESH_BINARY)

        # Calculate the angles of the detected rectangle and the perspective transform
        angle_detected = np.degrees(np.arctan2(strongest_rectangle[1][0, 1] - strongest_rectangle[0][0, 1],
                                               strongest_rectangle[1][0, 0] - strongest_rectangle[0][0, 0]))
        angle_matrix = np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))
        angle_sum = angle_matrix + angle_detected
        
        # condition for rotating the result to keep displayed perspective
        if angle_sum >= 90 and angle_matrix <= -45:
            binary_result = cv2.rotate(binary_result, cv2.ROTATE_90_CLOCKWISE)

        # Display the original frame with contours
        cv2.imshow('Webcam with Closed Rectangles', frame_contours)

        # Display the binary warped image
        cv2.imshow('Warped Image', binary_result)
    else:
        # Display the original frame with contours
        cv2.imshow('Webcam with Closed Rectangles', frame_contours)

    # Break the loop if esc key is pressed
    if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ASCII code for the escape key
        break

# Release the capture when done
cap.release()
cv2.destroyAllWindows()