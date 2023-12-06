import cv2
import numpy as np

def process_frame(frame, lower_bound, upper_bound):
    # Create a mask for colors within the specified range
    mask = cv2.inRange(frame, lower_bound, upper_bound)

    # Bitwise AND to keep colors within the specified range
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Invert the mask to get the region outside the specified range
    inverse_mask = cv2.bitwise_not(mask)

    # Create a color gradient for the background
    gradient = np.zeros_like(frame)
    gradient[:, :, 0] = np.linspace(0, 255, frame.shape[0])[:, np.newaxis]
    gradient[:, :, 1] = np.linspace(0, 255, frame.shape[0])[:, np.newaxis]
    gradient[:, :, 2] = np.linspace(0, 255, frame.shape[0])[:, np.newaxis]

    # Bitwise AND to keep the gradient in the region outside the specified range
    result_bg = cv2.bitwise_and(gradient, gradient, mask=inverse_mask)

    # Combine the foreground and background
    result = cv2.add(result, result_bg)

    # Convert the result to grayscale
    result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    return result_gray

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    lower_bound = np.array([200, 200, 200], dtype=np.uint8)
    upper_bound = np.array([255, 255, 255], dtype=np.uint8)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        processed_frame = process_frame(frame, lower_bound, upper_bound)

        cv2.imshow('Processed Frame (Color Range)', processed_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()