import cv2
import numpy as np
import time
import argparse
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense
from keras.optimizers import Adam, rmsprop
import sys
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('test.avi', fourcc, 20.0, (640,480))
while(cap.isOpened()):
    ret, frame = cap.read()
    out.write(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

#동영상 저장 파일

cap = cv2.VideoCapture("test.avi")

# 옵션 설명 http://layer0.authentise.com/segment-background-using-computer-vision.html
fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=100)


while(1):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)



    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(fgmask)


    for index, centroid in enumerate(centroids):
        if stats[index][0] == 0 and stats[index][1] == 0:
            continue
        if np.any(np.isnan(centroid)):
            continue


        x, y, width, height, area = stats[index]
        centerX, centerY = int(centroid[0]), int(centroid[1])

        if area > 100:
            cv2.circle(frame, (centerX, centerY), 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255))


    cv2.imshow('mask',fgmask)
    cv2.imshow('frame',frame)

cap.release()
cv2.destroyAllWindows()


#동영상 인식 파일


















test_image = mpimg.imread(os.path.join("predictions.png"))
plt.imshow(test_image)
plt.axis("on")
plt.show()