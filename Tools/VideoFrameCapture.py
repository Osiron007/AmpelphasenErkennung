
import cv2 as cv

import time

cap = cv.VideoCapture("/home/dlm/PycharmProjects/AmpelphasenErkennung/Videos/Vid_08_12.avi")

frameCnt = 0

ret = True

while(ret):
    ret, frame = cap.read()

    #cv.imshow("MyFrame", frame)
    frameCnt = frameCnt + 1
    cv.imwrite("/home/dlm/PycharmProjects/AmpelphasenErkennung/Bilder/NewData/Frame%d.jpg" % frameCnt, frame)


print("Anz Frames: " + str(frameCnt))
cap.release()
cv.destroyAllWindows()




