import numpy as np
from pyModules.Utilities import Line
from pyModules.Utilities import visualize_lane


def find_polynom_from_former(warped_image, left_line: Line, right_line: Line):

    # Define Margin for search window
    margin = 100

    # Get the none-zero pixels from the picture
    none_zero_xy = warped_image.nonzero()
    # Extract the y & x values for polynom calculation
    none_zero_y = np.array(none_zero_xy[0])
    none_zero_x = np.array(none_zero_xy[1])

    # Calculate the x none-zeros based on former polynom
    # In the first case there was none so we had to do it via histograms and parts in the picture
    left_line_x_none_zeros = left_line.average_coefficient[0] * (none_zero_y ** 2) + \
                             left_line.average_coefficient[1] * none_zero_y + left_line.average_coefficient[2]

    right_line_x_none_zeros = right_line.average_coefficient[0] * (none_zero_y ** 2) + \
                              right_line.average_coefficient[1] * none_zero_y + right_line.average_coefficient[2]

    left_good_indices = ((none_zero_x > (left_line_x_none_zeros - margin)) &
                         (none_zero_x < (left_line_x_none_zeros + margin)))
    right_good_indices = ((none_zero_x > (right_line_x_none_zeros - margin)) &
                          (none_zero_x < (right_line_x_none_zeros + margin)))

    left_line_x = none_zero_x[left_good_indices]
    left_line_y = none_zero_y[left_good_indices]
    right_line_x = none_zero_x[right_good_indices]
    right_line_y = none_zero_y[right_good_indices]

    # Create new coefficients for the current frame
    # We use y over x since the line might be almost vertical. ::> np.polyfit(y, x)
    left_lane_new_coefficient = np.polyfit(left_line_y, left_line_x, 2)
    right_lane_new_coefficient = np.polyfit(right_line_y, right_line_x, 2)

    # create y values
    y_values = np.linspace(0, warped_image.shape[0]-1, warped_image.shape[0])

    # Calculate the new x_values based on the new coefficients
    left_lane_x = left_lane_new_coefficient[0] * y_values ** 2 + \
        left_lane_new_coefficient[1] * y_values + left_lane_new_coefficient[2]
    right_lane_x = right_lane_new_coefficient[0] * y_values ** 2 + \
        right_lane_new_coefficient[1] * y_values + right_lane_new_coefficient[2]

    out_img = visualize_lane(warped_image, left_line_x, left_line_y, right_line_x, right_line_y, left_lane_x,
                             right_lane_x, y_values)

    return out_img, left_lane_new_coefficient, right_lane_new_coefficient, y_values, left_lane_x, right_lane_x
