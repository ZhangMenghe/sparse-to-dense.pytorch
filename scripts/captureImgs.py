import time
import numpy as np
import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")
id=0
while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()

	# Our operations on the frame come here
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Display the resulting frame
	cv2.imshow('frame',gray)
	res = cv2.waitKey(33)
	if res==27:
		print("save img")
		cv2.imwrite('bow/'+str(id)+'.jpg',gray)
		id+=1
cap.release()
cv2.destroyAllWindows()

