import numpy as np
import cv2
from pyModules.Utilities import visualize_lane


def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom quarter of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines

    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 10
    # Set the width of the windows +/- margin
    margin = 50
    # Set minimum number of pixels found to recenter window
    minpix = 100

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // nwindows)
    # Identify the x and y positions of all none_zero_xy pixels in the image
    none_zero_xy = binary_warped.nonzero()
    non_zero_y = np.array(none_zero_xy[0])
    none_zero_x = np.array(none_zero_xy[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        # Update the boundaries of the window
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Identify the none_zero_xy pixels in x and y within the window
        good_left_inds = ((non_zero_y >= win_y_low) & (non_zero_y < win_y_high) &
                          (none_zero_x >= win_xleft_low) & (none_zero_x < win_xleft_high)).nonzero()[0]
        good_right_inds = ((non_zero_y >= win_y_low) & (non_zero_y < win_y_high) &
                           (none_zero_x >= win_xright_low) & (none_zero_x < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # Recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(none_zero_x[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(none_zero_x[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = none_zero_x[left_lane_inds]
    lefty = non_zero_y[left_lane_inds]
    rightx = none_zero_x[right_lane_inds]
    righty = non_zero_y[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Get second order polynomial for the found coordinates
    left_polynom = np.polyfit(lefty, leftx, 2)
    right_polynom = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    y_values = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    # print(binary_warped.shape[0])
    try:
        left_fitx = left_polynom[0] * y_values ** 2 + left_polynom[1] * y_values + left_polynom[2]
        right_fitx = right_polynom[0] * y_values ** 2 + right_polynom[1] * y_values + right_polynom[2]
    except TypeError:
        # Avoids an error if `left_fitx` and `right_fitx` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1 * y_values ** 2 + 1 * y_values
        right_fitx = 1 * y_values ** 2 + 1 * y_values

    out_img = visualize_lane(binary_warped, leftx, lefty, rightx, righty, left_fitx, right_fitx, y_values)

    return out_img, left_polynom, right_polynom, y_values, left_fitx, right_fitx
