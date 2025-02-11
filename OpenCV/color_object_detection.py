import cv2 as cv
from PIL import Image
from utils_color import get_limits

yellow = [0, 255, 255]  # Placeholder for yellow (BGR value, not directly used)
cam = cv.VideoCapture(0)

while True:
    ret, frame = cam.read()

    if not ret:
        print("Failed to grab frame.")
        break

    # Convert the frame to HSV color space
    hsvImage = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Get HSV limits for yellow
    lowerLimit, upperLimit = get_limits(color=yellow)

    # Create a mask for yellow color
    mask = cv.inRange(hsvImage, lowerLimit, upperLimit)

    # Create bounding box
    mask_ = Image.fromarray(mask)
    bbox = mask_.getbbox()
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        frame = cv.rectangle(frame, (x1, y1), (x2, y2), [0, 255, 0], 3)

    # Display the video feed
    cv.imshow('video-frame', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv.destroyAllWindows()
