import numpy as np
import cv2
import glob

W = 9
H = 6

def calibrate():
	# termination criteria
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

	# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
	objp = np.zeros((W*H,3), np.float32)
	objp[:,:2] = np.mgrid[0:H,0:W].T.reshape(-1,2)

	# Arrays to store object points and image points from all the images.
	objpoints = [] # 3d point in real world space
	imgpoints = [] # 2d points in image plane.

	images = glob.glob('imgs/*.jpg')

	for fname in images:
		print(fname)
		img = cv2.imread(fname)
		#img = cv2.resize(img,(1600,900))
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

		# gray=cv2.resize(gray,(1600,900))
		gray=cv2.resize(gray,(640,480))

		# Find the chess board corners
		ret, corners = cv2.findChessboardCorners(gray, (H,W),None)
		print(ret)
		# If found, add object points, image points (after refining them)
		if ret == True:
			objpoints.append(objp)

			corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
			imgpoints.append(corners2)

			# Draw and display the corners
			img = cv2.drawChessboardCorners(img, (H,W), corners2,ret)
			#cv2.imshow('img',img)
			#cv2.waitKey(500)

	#cv2.destroyAllWindows()
	# Calibrate the camera and save the results
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
	np.savez('calib.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
	
	# Print the camera calibration error
	error = 0

	for i in range(len(objpoints)):
		imgPoints, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
		error += cv2.norm(imgpoints[i], imgPoints, cv2.NORM_L2) / len(imgPoints)

	print("Total error: ", error / len(objpoints))


def modify_calibrate():
	file= 'calib.npz'
	data = np.load(file)
	mtx,dist,rvecs,tvecs=data['mtx'],data['dist'],data['rvecs'],data['tvecs'],

	# Load one of the test images
	img = cv2.imread('sample_chessboard/left12.jpg')
	h, w = img.shape[:2]

	# Obtain the new camera matrix and undistort the image
	newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
	undistortedImg = cv2.undistort(img, mtx, dist, None, newCameraMtx)

	# Crop the undistorted image
	# x, y, w, h = roi
	# undistortedImg = undistortedImg[y:y + h, x:x + w]

	# Display the final result
	cv2.imshow('chess board', np.hstack((img, undistortedImg)))
	cv2.waitKey(0)
	cv2.destroyAllWindows()
calibrate()	
#modify_calibrate()
