# # Advanced Lane Finding Project
#
# The goals / steps of this project are the following:
#
# * Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
# * Apply a distortion correction to raw images.
# * Use color transforms, gradients, etc., to create a threshold binary image.
# * Apply a perspective transform to rectify binary image ("birds-eye view").
# * Detect lane pixels and fit to find the lane boundary.
# * Determine the curvature of the lane and vehicle position with respect to center.
# * Warp the detected lane boundaries back onto the original image.
# * Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
#
# First, I import all the required libraries
import cv2
import matplotlib.pyplot as plt

# import the other files
from pyModules.CalibrateCamera import calibrate_camera
from pyModules.CalibrateCamera import undistort_image

from pyModules.ColorSpace import combine_thresholds

from pyModules.PerspectiveTransform import warp_picture
from pyModules.PerspectiveTransform import get_matrixes

from pyModules.LineFit import fit_polynomial
from pyModules.Utilities import measure_curvature_real
from pyModules.Utilities import print_on_image
from pyModules.Utilities import offset_calculation

# calibrate Camera:
mtx_camera, dist_camera = calibrate_camera()

# Load Image for testing
# Get Image and undistort Image
img = cv2.imread('test_images/straight_lines1.jpg')
img_undist = undistort_image(img, mtx_camera, dist_camera)[..., ::-1]
combined_img = combine_thresholds(img_undist)

img_size = (img.shape[1], img.shape[0])
mtx_persp, mtx_persp_inv = get_matrixes(img_size)

warped = warp_picture(combined_img, mtx_persp)

# Detect the Lines in the warped Image
out_img, left_poly, right_poly, poly, leftx, rightx = fit_polynomial(warped)

# Weight the curvature ratios based on the amount of pixels
left_r, right_r = measure_curvature_real(poly, left_poly, right_poly)
weighted_r = (((left_r * leftx.shape[0]) + (right_r * rightx.shape[0])) / (leftx.shape[0] + rightx.shape[0]))
offset = offset_calculation(leftx, rightx, img_size)

# Print the lines back into the image:
lines_img = warp_picture(out_img, mtx_persp_inv)
output_img = cv2.addWeighted(img_undist, 1, lines_img, 0.9, 1)
print_on_image(output_img, weighted_r, offset)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.imshow(img_undist)
ax1.set_title('Undistored Image', fontsize=30)
ax2.imshow(output_img)
ax2.set_title('Result Image', fontsize=30)
plt.show()
