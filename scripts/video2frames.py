import time
import numpy as np
import cv2
import sys, getopt

def main(argv):
    try:
      opts, args = getopt.getopt(argv,"hi:o:",["ifile="])
    except getopt.GetoptError:
      print ('video2frames.py -i <inputfile>')
      sys.exit(2)
    
    inputfile = ""

    for opt, arg in opts:
        if opt in ("-i", "--ifile"):
            inputfile = arg
    out_dir = "_frames/"
    cap = cv2.VideoCapture(inputfile)
    if not cap.isOpened():
        raise IOError("Cannot open video ")
    
    ret, frame = cap.read()
    
    count = 0
    while(ret):
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(out_dir+str(count)+'.png',frame)
        # cv2.imshow('frame',frame)
        count+=1
        # Capture frame-by-frame
        ret, frame = cap.read()        
        # Display the resulting frame
        

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
   main(sys.argv[1:])