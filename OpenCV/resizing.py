import cv2 as cv
img = cv.imread('photos/w8.jpg')

resized_img = cv.resize(img, (700, 467))

print("Original size", img.shape)
cv.imshow('image frame', img)

print("New size", resized_img.shape)
cv.imshow('resized image frame', resized_img)
cv.waitKey(0)
