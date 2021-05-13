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
from pyModules.LineFit_from_line import find_polynom_from_former
from pyModules.Utilities import measure_curvature_real
from pyModules.Utilities import print_on_image
from pyModules.Utilities import offset_calculation
from pyModules.Utilities import Line

# calibrate Camera:
mtx_camera, dist_camera = calibrate_camera()

# Load Image for testing
# Get Image and undistort Image
img = cv2.imread('test_images/test4.jpg')
img_undist = undistort_image(img, mtx_camera, dist_camera)

# Output Undistorted Image
cv2.imwrite('output_images/undist_image.png', img_undist)

combined_img = combine_thresholds(img_undist)

# Output Combined Image
plt.imsave('output_images/combined_image.jpg', combined_img, cmap='gray')

img_size = (img.shape[1], img.shape[0])
mtx_persp, mtx_persp_inv = get_matrixes(img_size)

img_warped = warp_picture(combined_img, mtx_persp)
plt.imsave('output_images/warped_image.jpg', img_warped, cmap='gray')

left_line = Line()
right_line = Line()
# Detect the Lines in the warped Image
out_img, left_poly, right_poly, poly, leftx, rightx = fit_polynomial(img_warped)
left_line.available = True
left_line.calc_average(left_poly)
right_line.available = True
right_line.calc_average(right_poly)

print("Left Average: ", left_line.average_coefficient)
print("Left Coeffs: ", left_line.coefficients)
print("Right Average: ", right_line.average_coefficient)
print("Right Coeffs: ", right_line.coefficients)

# Weight the curvature ratios based on the amount of pixels
left_r, right_r = measure_curvature_real(poly, left_poly, right_poly)
weighted_r = (((left_r * leftx.shape[0]) + (right_r * rightx.shape[0])) / (leftx.shape[0] + rightx.shape[0]))
offset = offset_calculation(leftx, rightx, img_size)

# Print the lines back into the image:
lines_img = warp_picture(out_img, mtx_persp_inv)
output_img = cv2.addWeighted(img_undist, 1, lines_img, 0.9, 1)
print_on_image(output_img, weighted_r, offset)

f, (ax3, ax4) = plt.subplots(1, 2, figsize=(15, 5))
ax3.imshow(img_undist)
ax3.set_title('Undistored Image', fontsize=12)
ax4.imshow(output_img)
ax4.set_title('Result Image', fontsize=12)
plt.show()
