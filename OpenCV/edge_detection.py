import cv2 as cv

img = cv.imread('photos/w8.jpg')
img = cv.resize(img, (700, 467))
img2 = cv.imread('photos/tiger.jpg')
img2 = cv.resize(img2, (480, 320))

gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray_img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

canny = cv.Canny(gray_img, 100, 200)
canny2 = cv.Canny(gray_img2, 100, 200)
sobelx = cv.Sobel(gray_img2, ddepth=cv.CV_64F, dx=1, dy=0, ksize=3)
sobely = cv.Sobel(gray_img2, ddepth=cv.CV_64F, dx=0, dy=1, ksize=3)
sobelxy = cv.Sobel(gray_img2, ddepth=cv.CV_64F, dx=1, dy=1, ksize=3)


#cv.imshow('hellcat',img)
#cv.imshow('tiger',img)
#cv.imshow('gray hellcat',gray_img)
cv.imshow('gray tiger',gray_img2)
#cv.imshow('canny edge hellcat',canny)
cv.imshow('canny edge tiger',canny2)
cv.imshow('sobelx edge tiger',sobelx)
cv.imshow('sobely edge tiger',sobely)
cv.imshow('sobelxy edge tiger',sobelxy)
cv.waitKey(0)

