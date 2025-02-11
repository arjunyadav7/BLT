import cv2 as cv
import numpy as np

img = cv.imread('photos/birds.jpg')
img = cv.resize(img, (735,490))

img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(img_gray, 100, 255, cv.THRESH_BINARY_INV)

contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    #print(cv.contourArea(cnt))
    #cv.drawContours(img, cnt, contourIdx=-1, color=(0,255,0), thickness=1)
    x1, y1, w, h = cv.boundingRect(cnt)
    cv.rectangle(img, (x1,y1), (x1+w,y1+h), color=(0,255,0), thickness=2)

cv.imshow('birds', img)
cv.imshow('Thresh_BIN_INV', thresh)
cv.waitKey(0)
