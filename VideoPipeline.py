# Import cv2
import cv2
# import matplotlib.pyplot as plt

# import the other internal files
from pyModules.CalibrateCamera import calibrate_camera
from pyModules.CalibrateCamera import undistort_image

from pyModules.ColorSpace import combine_thresholds

from pyModules.PerspectiveTransform import warp_picture
from pyModules.PerspectiveTransform import get_matrixes

from pyModules.LineFit import fit_polynomial
from pyModules.LineFit_from_line import find_polynom_from_former
from pyModules.Utilities import measure_curvature_real
from pyModules.Utilities import offset_calculation
from pyModules.Utilities import print_on_image
from pyModules.Utilities import Line

# calibrate Camera:
mtx_camera, dist_camera = calibrate_camera()
right_line = Line()  # not sure how to use yet
left_line = Line()  # not sure how to use yet

# image = cv2.imread('test_images/test4.jpg')
counter = [0, 0]

def process_image(image):

    img_undist = undistort_image(image, mtx_camera, dist_camera)
    combined_img = combine_thresholds(img_undist)

    img_size = (image.shape[1], image.shape[0])
    mtx_persp, mtx_persp_inv = get_matrixes(img_size)

    warped = warp_picture(combined_img, mtx_persp)

    # Detect the Lines in the warped Image
    if (left_line.available and right_line.available):
        out_img, left_poly, right_poly, poly, leftx, rightx = find_polynom_from_former(warped, left_line, right_line)
        counter[1] += 1
    else:
        out_img, left_poly, right_poly, poly, leftx, rightx = fit_polynomial(warped)
        left_line.available = True
        right_line.available = True
        counter[0] += 1

    left_line.calc_average(left_poly)
    right_line.calc_average(right_poly)


    # Getting Curvature for left and right side
    left_r, right_r = measure_curvature_real(poly, left_poly, right_poly)
    # Weight the curvature ratios based on the amount of pixels
    weighted_r = (((left_r * leftx.shape[0]) + (right_r * rightx.shape[0])) / (leftx.shape[0] + rightx.shape[0]))

    offset = offset_calculation(leftx, rightx, img_size)
    lines_img = warp_picture(out_img, mtx_persp_inv)

    # Print the lines back into the image:
    print_on_image(img_undist, weighted_r, offset)
    output_img = cv2.addWeighted(img_undist, 1, lines_img, 0.9, 0)

# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
# ax1.imshow(image)
# ax1.set_title('Undistored Image', fontsize=12)
# ax2.imshow(output_img)
# ax1.set_title('Output Image', fontsize=12)
# plt.show()
    return output_img

from moviepy.editor import VideoFileClip
from IPython.display import HTML
from pyModules.LineFit_from_line import find_polynom_from_former

output = 'project_video_output_short.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(output, audio=False)

print(counter)

