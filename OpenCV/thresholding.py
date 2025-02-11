import cv2 as cv
img = cv.imread('photos/w8.jpg')
img = img[250:700, 300:1100]
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

ret, simple_thresh = cv.threshold(gray_img, 90, 255, cv.THRESH_BINARY)
adaptive_thresh = cv.adaptiveThreshold(gray_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 3, 5)
#otsu_thresh = cv.threshold(gray_img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

cv.imshow('original img', img)
cv.imshow('simple thresh', simple_thresh)
cv.imshow('adaptive thresh', adaptive_thresh)
#cv.imshow('otsu thresh', otsu_thresh)
cv.waitKey(0)
