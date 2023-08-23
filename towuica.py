import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Flip the image horizontally to remove the mirror effect
    frame = cv2.flip(frame, 1)

    # Get the frame size
    frame_height, frame_width, _ = frame.shape
    # print("Frame Size:", frame_width, "x", frame_height)

    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and details
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply simple thresholding to the blurred image
    # _, thresholded = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)

     # Apply adaptive thresholding to the blurred image
    thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 5)

     # Apply Otsu's thresholding to the blurred image
    # _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply Canny edge detection to the thresholded image
    edges = cv2.Canny(thresholded, 30, 150)

    # Show the original image with the edges and lines detected
    cv2.imshow('original', frame)
    cv2.imshow('Edges', edges)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()