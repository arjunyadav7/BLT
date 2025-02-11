import cv2 as cv
img = cv.imread('photos/w8.jpg')

img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
img_bgra = cv.cvtColor(img, cv.COLOR_BGR2BGRA)

cv.imshow('Original BGR', img)
cv.imshow('RGB', img_rgb)
cv.imshow('GRAY', img_gray)
cv.imshow('HSV', img_hsv)
cv.imshow('BGRA', img_bgra)
cv.waitKey(0)
