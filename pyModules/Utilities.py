import numpy as np
import cv2
import math

# Define conversions in x and y from pixels space to meters
# Parameters for real world transform
ym_per_pix = 30 / 720  # meters per pixel in y dimension
xm_per_pix = 3.7 / 800  # meters per pixel in x dimension


class Line:
    def __init__(self):
        self.available = False
        self.average_coefficient = None
        self.coefficients = []

    def calc_average(self, coefficient):
        self.coefficients.append(coefficient)
        if len(self.coefficients) > 5:
            self.coefficients = self.coefficients[1:6]


def measure_curvature_real(ploty, left_radius, right_radius):
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Calculation of R_curve
    left_curve_radius = (((1 + (2 * left_radius[0] * y_eval * ym_per_pix +
                                left_radius[1]) ** 2) ** (3 / 2)) / np.abs(2 * left_radius[0]))
    right_curve_radius = (((1 + (2 * right_radius[0] * y_eval * ym_per_pix +
                                 right_radius[1]) ** 2) ** (3 / 2)) / np.abs(2 * right_radius[0]))

    return left_curve_radius, right_curve_radius


def offset_calculation(leftx, rightx, img_size):
    position = (np.average(leftx) + np.average(rightx)) / 2
    offset = ((img_size[0] / 2) - position) * xm_per_pix

    return offset


def print_on_image(image, radius, offset):
    # Parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 255, 255)

    radius = math.trunc(radius * 100) / 100
    radius_str = "The Curve radius is: " + str(radius) + 'm'

    offset = math.trunc(offset * 100) / 100
    offset_str = "Offset: " + str(offset) + 'm'

    cv2.putText(image, radius_str, (10, 50), font, 1.2, color, 2, cv2.LINE_AA)
    cv2.putText(image, offset_str, (10, 100), font, 1.2, color, 2, cv2.LINE_AA)

    return image
