import cv2
import numpy as np

def process_frame(frame):
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define a color range for detecting skin color (you may need to adjust these values)
    lower_red = np.array([0, 100, 100], dtype=np.uint8)
    upper_red = np.array([10, 255, 255], dtype=np.uint8)

    # Create a binary mask using the inRange function
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Perform morphological operations to remove noise
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Apply the mask to the original frame
    result = cv2.bitwise_and(frame, frame, mask=mask)

    return result

def main():
    # Open a connection to the webcam (usually 0 for built-in webcams)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Process the frame to display only skin
        processed_frame = process_frame(frame)

        # Display the original and processed frames
        cv2.imshow('Original Frame', frame)
        cv2.imshow('Processed Frame', processed_frame)

        # Break the loop when 'Esc' key is pressed
        if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ASCII code for 'Esc' key
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
