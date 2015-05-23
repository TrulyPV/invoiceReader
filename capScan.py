#!/usr/bin/env python
#-*-coding:utf-8-*-

import time
import cv2
import numpy as np


def getXY(box):
    avrx = 0
    avry = 0
    for i in box:
        avrx += i[0]
        avry += i[1]
    minx,maxx = (0,0)
    miny,maxy = (0,0)
    cntp = len(box)
    avrx /= cntp
    avry /= cntp
    
    for i in box:
        
        if i[0] < avrx:
            minx += i[0]
        else:
            maxx += i[0]

        if i[1] < avry:
            miny += i[1]
        else:
            maxy += i[1]
    return (minx/2,maxx/2),(miny/2,maxy/2)

cap = cv2.VideoCapture(1);

cap.set(3,1280)

cap.set(4,1024)

# time.sleep(2)

# cap.set(15, -8.0)

while True:
    ret, img = cap.read()
    # cv2.imshow("input", img)

    # load the image and convert it to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    equ = cv2.equalizeHist(gray)
    res = np.hstack((gray,equ))
    # cv2.imshow('equalized',res)
    #stacking images side-by-side

    # compute the Scharr gradient magnitude representation of the images
    # in both the x and y direction
    gradX = cv2.Sobel(res, ddepth = cv2.cv.CV_32F, dx = 1, dy = 0, ksize = -1)
    gradY = cv2.Sobel(res, ddepth = cv2.cv.CV_32F, dx = 0, dy = 1, ksize = -1)

    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    # cv2.imshow('gradientedX',gradient)

    gradient = cv2.convertScaleAbs(gradient)

    # cv2.imshow('gradientedXY',gradient)

    # gray = np.float32(res)

    # # 输入图像必须是 float32 ,最后一个参数在 0.04 到 0.05 之间
    # dst = cv2.cornerHarris(gray,2,3,0.04)
    # #result is dilated for marking the corners, not important
    # dst = cv2.dilate(dst,None)
    # # Threshold for an optimal value, it may vary depending on the image.
    # img[dst>0.01*dst.max()]=[0,0,255]

    # blur and threshold the image
    blurred = cv2.blur(gradient, (9, 9))
    (_, thresh) = cv2.threshold(gray, 205, 255, cv2.THRESH_BINARY)

    cv2.imshow('threshed',thresh)

    # construct a closing kernel and apply it to the thresholded image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # perform a series of erosions and dilations
    closed = cv2.erode(closed, None, iterations = 4)
    closed = cv2.dilate(closed, None, iterations = 4)

    # find the contours in the thresholded image, then sort the contours
    # by their area, keeping only the largest one
    (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
     cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
    
    # compute the rotated bounding box of the largest contour
    rect = cv2.minAreaRect(c)

    box = np.int0(cv2.cv.BoxPoints(rect))
    # draw a bounding box arounded the detected barcode and display the
    # image
    img1=img.copy()
    cv2.drawContours(img1, [box], -1, (0, 255, 0), 3)
    cv2.imshow("Image", img1)

    



    key = cv2.waitKey(10) & 0x00ff
    if key  == 27:
        break
    elif key == ord('s'): # wait for 's' key to save and exit

        # row1 = ((box[1][0] + box[2][0])/2)
        # row2 = ((box[0][0] + box[3][0])/2)
        # col1 = ((box[2][1] + box[3][1])/2)
        # col2 = ((box[0][1] + box[1][1])/2)
        rows,cols = getXY(box) 
        
        roi = img[cols[0]:cols[1], rows[0]:rows[1]]
        
        fn = '%s.png' % time.strftime('%y%m%d-%H%M%S',time.localtime())
        cv2.imwrite(fn,roi)
        print 'saved to %s' % fn




cv2.destroyAllWindows() 
cv2.VideoCapture(1).release()