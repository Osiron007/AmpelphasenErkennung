
import cv2 as cv

import time

cap = cv.VideoCapture(0)

fourcc = cv.VideoWriter_fourcc(*'XVID')
output = cv.VideoWriter('Videos/AmpelVid9.avi',fourcc, 20.0, (640,480))

start_time = time.time()
print(time.time())

while(cap.isOpened):
    ret, frame = cap.read()
    cv.imshow("Vid", frame)

    output.write(frame)

    if time.time() > start_time+60:
        break

#cv.imshow("Ampel_Rot",image)
#cv.imshow("Ampel_Rot_graustufen",gray)
#cv.waitKey(0)
print(time.time())
cap.release()
output.release()
cv.destroyAllWindows()




