import time
import cv2
import numpy as np
import os

# initialize the camera and grab a reference to the raw camera capture
camera = cv2.VideoCapture(0) 

# allow the camera to warmup
time.sleep(0.1)

# capture frames from the camera
while (True):
    ret, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image = frame
    # we do something here
    # we get the image or something then run some matching
    # if we get a match, we draw a square on it or something
    img_rbg = image
    template = cv2.imread(os.getcwd() + r"\GTA_V_Casino_Heist_Fingerprint_diagnosis\images\1\Screenshot_1.crop.png", 0)
    w, h = template.shape[::-1]

    res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)

    threshold = 0.6

    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(image, (pt[0], pt[1]), (pt[0] + w, pt[1] + h),
              (0, 0, 255), 2)
        

    # show the frame
    cv2.imshow("Frame", img_rbg)
    
    # if the `q` key was pressed, break from the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break