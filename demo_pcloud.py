# Package importation
import numpy as np
import cv2
import glob
from plyfile import PlyData, PlyElement


# Filtering
kernel= np.ones((3,3),np.uint8)

def coords_mouse_disp(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        average=0
        for u in range (-1,2):
            for v in range (-1,2):
                average += disp[y+u,x+v]
        average=average/9
        Distance= -593.97*average**(3) + 1506.8*average**(2) - 1373.1*average + 522.06
        Distance= np.around(Distance*0.01,decimals=2)
        if Distance <= 0.5:
            print('Warning --- Distance: '+ str(Distance)+' m')
        else:
            print('Distance: '+ str(Distance)+' m')

#*************************************************
#***** Parameters for Distortion Calibration *****
#*************************************************

# Termination criteria
criteria =(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points
objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all images
objpoints= []   # 3d points in real world space
imgpointsR= []   # 2d points in image plane
imgpointsL= []

imagesLeft = glob.glob('images/stereoLeft/*.png')
imagesRight = glob.glob('images/stereoRight/*.png')

# Start calibration from the camera
print('Starting calibration for the 2 cameras... ')
# Call all saved images
for imgLeft, imgRight in zip(imagesLeft, imagesRight):
    ChessImaL = cv2.imread(imgLeft)
    ChessImaR = cv2.imread(imgRight)
    ChessImaL = cv2.cvtColor(ChessImaL, cv2.COLOR_BGR2GRAY)
    ChessImaR = cv2.cvtColor(ChessImaR, cv2.COLOR_BGR2GRAY)
    retR, cornersR = cv2.findChessboardCorners(ChessImaR,
                                               (9,6),None)  # Define the number of chees corners we are looking for
    retL, cornersL = cv2.findChessboardCorners(ChessImaL,
                                               (9,6),None)  # Left side
    if (True == retR) & (True == retL):
        objpoints.append(objp)
        cv2.cornerSubPix(ChessImaR,cornersR,(11,11),(-1,-1),criteria)
        cv2.cornerSubPix(ChessImaL,cornersL,(11,11),(-1,-1),criteria)
        imgpointsR.append(cornersR)
        imgpointsL.append(cornersL)

# Determine the new values for different parameters
#   Right Side
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints,
                                                        imgpointsR,
                                                        ChessImaR.shape[::-1],None,None)
hR,wR= ChessImaR.shape[:2]
OmtxR, roiR= cv2.getOptimalNewCameraMatrix(mtxR,distR,
                                                   (wR,hR),1,(wR,hR))

#   Left Side
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints,
                                                        imgpointsL,
                                                        ChessImaL.shape[::-1],None,None)
hL,wL= ChessImaL.shape[:2]
OmtxL, roiL= cv2.getOptimalNewCameraMatrix(mtxL,distL,(wL,hL),1,(wL,hL))

print('Ready to process the images')

#********************************************
#***** Calibrate the Cameras for Stereo *****
#********************************************

# StereoCalibrate function
retS, MLS, dLS, MRS, dRS, R, T, E, F= cv2.stereoCalibrate(objpoints,
                                                          imgpointsL,
                                                          imgpointsR,
                                                          mtxL,
                                                          distL,
                                                          mtxR,
                                                          distR,
                                                          ChessImaR.shape[::-1],
                                                          criteria = criteria_stereo,
                                                          flags = cv2.CALIB_FIX_INTRINSIC)

# StereoRectify function
rectify_scale= 0 # if 0 image cropped, if 1 image nor cropped
RL, RR, PL, PR, Q, roiL, roiR= cv2.stereoRectify(MLS, dLS, MRS, dRS,
                                                 ChessImaR.shape[::-1], R, T,
                                                 rectify_scale,(0,0))  # last paramater is alpha, if 0= croped, if 1= not croped
# initUndistortRectifyMap function
Left_Stereo_Map= cv2.initUndistortRectifyMap(MLS, dLS, RL, PL,
                                             ChessImaR.shape[::-1], cv2.CV_16SC2)   # cv2.CV_16SC2 this format enables us the programme to work faster
Right_Stereo_Map= cv2.initUndistortRectifyMap(MRS, dRS, RR, PR,
                                              ChessImaR.shape[::-1], cv2.CV_16SC2)


#*******************************************
#***** Parameters for the StereoVision *****
#*******************************************

# Create StereoSGBM and prepare all parameters
window_size = 3
min_disp = 2
num_disp = 130-min_disp
stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
    numDisparities = num_disp,
    blockSize = window_size,
    uniquenessRatio = 10,
    speckleWindowSize = 100,
    speckleRange = 32,
    disp12MaxDiff = 5,
    P1 = 8*3*window_size**2,
    P2 = 32*3*window_size**2)

# Used for the filtered image
stereoR=cv2.ximgproc.createRightMatcher(stereo) # Create another stereo for right this time

# WLS FILTER Parameters
lmbda = 80000
sigma = 1.8
visual_multiplier = 1.0
 
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

#*************************************
#***** Image processing **************
#*************************************

# Load the left and right images
left_image = cv2.imread('left_image.png')
right_image = cv2.imread('right_image.png')

# Rectify the images on rotation and alignment
Left_nice= cv2.remap(left_image,Left_Stereo_Map[0],Left_Stereo_Map[1], interpolation = cv2.INTER_LANCZOS4, borderMode = cv2.BORDER_CONSTANT)
# Rectify the image using the calibration parameters founds during the initialization
Right_nice= cv2.remap(right_image,Right_Stereo_Map[0],Right_Stereo_Map[1], interpolation = cv2.INTER_LANCZOS4, borderMode = cv2.BORDER_CONSTANT)

# Convert from color(BGR) to gray
grayR= cv2.cvtColor(Right_nice,cv2.COLOR_BGR2GRAY)
grayL= cv2.cvtColor(Left_nice,cv2.COLOR_BGR2GRAY)

# Compute the 2 images for the Depth_image
disparity = stereo.compute(grayL,grayR)
# disparity_map = stereo.compute(grayL,grayR)

# Normalize the disparity map to [0, 255]
disparitynorm = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

# Load the left image in color
color_image = cv2.imread('left_image.png')

# Create an empty array for the 3D points
points_3D = np.zeros((color_image.shape[0] * color_image.shape[1], 3), dtype=np.float32)

# Create an empty array for the colors
colors = np.zeros((color_image.shape[0] * color_image.shape[1], 3), dtype=np.uint8)

# Fill the 3D points and colors arrays
i = 0
for y in range(color_image.shape[0]):
    for x in range(color_image.shape[1]):
        points_3D[i] = [x, y, disparitynorm[y,x]]
        colors[i] = color_image[y,x]
        i += 1

# Create a vertex element from the 3D points and colors arrays
vertex = np.array([tuple(points_3D[i]) + tuple(colors[i]) for i in range(points_3D.shape[0])], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

# Create a PlyElement from the vertex element
el = PlyElement.describe(vertex, 'vertex')

# Write the point cloud data to a .ply file
PlyData([el], text=True).write('point_cloud.ply')