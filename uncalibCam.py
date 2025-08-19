import numpy as np
import cv2

# Load the two images
img_left = cv2.imread('unCalibLeft.png')
img_right = cv2.imread('unCalibRight.png')

# Define the stereo parameters
focal_length = 1400 # in pixels
baseline = 0.13 # in meters
Q = np.array([[1, 0, 0, -0.5 * img_left.shape[1]],
              [0, 1, 0, -0.5 * img_left.shape[0]],
              [0, 0, 0, focal_length],
              [0, 0, -1/baseline, 0]])

# Compute the disparity map
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY))

# Compute the depth map
depth = cv2.reprojectImageTo3D(disparity, Q)
depth_map = depth[:,:,2] / 1000.0

# Display the depth map
cv2.imshow('Depth Map', depth_map)
cv2.waitKey(0)
cv2.destroyAllWindows()
