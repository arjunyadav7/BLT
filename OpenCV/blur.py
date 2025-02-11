import cv2 as cv
img = cv.imread('photos/w8.jpg')

k_size = 11
img_blur = cv.blur(img, (k_size,k_size))
img_gaussian_blur = cv.GaussianBlur(img, (k_size,k_size), 5)
img_median_blur = cv.medianBlur(img, k_size)

cv.imshow('img', img)
cv.imshow('simple blur img', img_blur)
cv.imshow('gaussian blur img', img_gaussian_blur)
cv.imshow('median blur img', img_median_blur)
cv.waitKey(0)
