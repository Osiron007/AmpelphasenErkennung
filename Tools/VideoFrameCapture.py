
import cv2 as cv

import time

cap = cv.VideoCapture("Videos/AmpelVid9.avi")

frameCnt = 0

ret = True

while(ret):
    ret, frame = cap.read()

    #cv.imshow("MyFrame", frame)
    frameCnt = frameCnt + 1
    cv.imwrite("Bilder/NewData/Frame%d.jpg" % frameCnt, frame)


print("Anz Frames: " + str(frameCnt))
cap.release()
cv.destroyAllWindows()




