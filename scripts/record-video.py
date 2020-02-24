import cv2
#Capture video from webcam
vid_capture = cv2.VideoCapture(0)
vid_cod = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter("cam_video.mp4", vid_cod, 20.0, (640,480))
startsaving = False
while(True):
     # Capture each frame of webcam video
     ret,frame = vid_capture.read()
     cv2.imshow("My cam video", frame)
     if(startsaving):
        output.write(frame)
     # Close and break the loop after pressing "x" key
     key = cv2.waitKey(1)
     if(key& 0XFF == ord('x')):
        break
     elif(key & 0XFF == ord('s')):
        startsaving = True 
# close the already opened camera
vid_capture.release()
# close the already opened file
output.release()
# close the window and de-allocate any associated memory usage
cv2.destroyAllWindows()