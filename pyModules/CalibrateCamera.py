import glob
import cv2
import numpy as np


def calibrate_camera():
    # Load all names of pictures
    images = glob.glob('camera_cal/calibration*.jpg')

    # Variables for Chess Board:
    corners_x = 9
    corners_y = 6
    # Set up Arrays for Imagepoints and Objectpoints
    img_pts = [] #2-D Picture Points
    obj_pts = [] #3-D Real World Point

    #Create the base for the object points
    obj_p = np.zeros((corners_y * corners_x, 3), np.float32)
    obj_p[:, :2] = np.mgrid[0:corners_x, 0:corners_y].T.reshape(-1, 2)

    for image in images:
        # Read in the image
        img = cv2.imread(image)

        # Convert to Gray Scale
        gray_chess = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find corners of Chess Board
        ret, corners = cv2.findChessboardCorners(gray_chess, (corners_x, corners_y), None)

        # Add detected corners and image points to the defined arrays
        if (ret == True):
            img_pts.append(corners)
            obj_pts.append(obj_p)

            # Draw and corners on image
            cv2.drawChessboardCorners(img, (corners_x, corners_y), corners, ret)

    # Get the Matrix and distortian coefficients and return them.
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, gray_chess.shape[::-1], None, None)
    return (mtx, dist)

# Undistort the image
# I devided it, so that not with every distortian the camera needs to be calibrated.


def undistort_image(img, mtx, dist):
    dst_img = cv2.undistort(img, mtx, dist, None, mtx)
    return dst_img
