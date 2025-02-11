import cv2 as cv
img = cv.imread('photos/w8.jpg')

img = img[250:700, 300:1100]

cv.imshow('crop frame', img)
cv.waitKey(0)
