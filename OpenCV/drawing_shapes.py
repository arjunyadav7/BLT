import cv2 as cv
import numpy as np

img = cv.imread('photos/birds.jpg')
img = cv.resize(img, (735,490))
print(img.shape)

# Define the points as a NumPy array with integer values
points = np.array([
    (200, 200), (220, 220), (240, 240), (260, 240),
    (280, 220), (300, 200), (280, 180), (260, 160),
    (240, 160), (220, 180)
], np.int32)
points = points.reshape((-1, 1, 2))

cv.line(img, (50, 100), (100, 400), (255, 0, 0), 2)
cv.rectangle(img, (100, 20), (300, 120), (155, 100, 160), -1)
cv.circle(img, (400, 100), 50, (50, 50, 150), 3)

cv.polylines(img, [points], isClosed=True, color=(0,120,0), thickness=5)
text = "Hello"
cv.putText(img, text, (400,400), fontFace=cv.FONT_HERSHEY_COMPLEX, color=(0,200,200), fontScale=3, thickness=2)

cv.imshow('drawing', img)
cv.waitKey(0)
