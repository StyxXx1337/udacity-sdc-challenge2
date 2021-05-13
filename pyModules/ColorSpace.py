import cv2
import numpy as np
import matplotlib.pyplot as plt

# =============================
# Parameters for Sobel
sX_kernel = 11
sY_kernel = 11
sMag_kernel = 21
sDir_kernel = 31
sX_thresh = (100, 255)
sY_thresh = (80, 255)
sMag_thresh = (80, 255)
sDir_thresh = (0.75, 1.10)
hls_thresh = (100, 255)


# binary S-Space output
def hls_Sspace(img_undist, thresh=(0, 255)):
    # Convert to HLS color space
    hls_img = cv2.cvtColor(img_undist, cv2.COLOR_RGB2HLS)
    # Apply a threshold to the S channel
    S_img = hls_img[:, :, 2]
    # Return a binary image of threshold result
    S_bin = np.zeros_like(S_img)
    S_bin[(S_img > thresh[0]) & (S_img <= thresh[1])] = 1

    return S_bin


# Sobel Functions
def abs_sobel_thresh(img_undist, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Convert to grayscale
    gray_img = cv2.cvtColor(img_undist, cv2.COLOR_RGB2GRAY)
    # Take the derivative in x or y given orient = 'x' or 'y'
    # only take the one required to improve performance
    if orient == 'x':
        sobel = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient == 'y':
        sobel = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    else:
        print("Error: Wrong Orientation")  # To be changed to Error Handler

    # Get the absolute value of the sobel
    sobel_dir_abs = np.absolute(sobel)
    # Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * sobel_dir_abs / np.max(sobel_dir_abs))
    # Create a binary mask
    sobel_binary = np.zeros_like(scaled_sobel)
    # Make 1's where the sobel is within the thresholds
    sobel_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return sobel_binary


def mag_thresh(img_undist, sobel_kernel=3, magn_thresh=(0, 255)):
    # Convert to grayscale
    gray_img = cv2.cvtColor(img_undist, cv2.COLOR_RGB2GRAY)
    # Get the gradient in x and y
    sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the magnitude
    sobel_mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    # Scale to 8-bit and convert to uint8 type
    sobel_scaled = np.uint8(255 * sobel_mag / np.max(sobel_mag))
    # Create a binary mask
    sobel_mag_binary = np.zeros_like(sobel_scaled)
    # Make 1's where the sobel_mag is within the thresholds
    sobel_mag_binary[(sobel_scaled >= magn_thresh[0]) & (sobel_scaled <= magn_thresh[1])] = 1

    return sobel_mag_binary


def dir_threshold(img_undist, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Convert to grayscale
    gray_img = cv2.cvtColor(img_undist, cv2.COLOR_RGB2GRAY)
    # Get the gradient in x and y
    sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Get the absolute value of the x and y gradients
    sobelx_abs = np.absolute(sobelx)
    sobely_abs = np.absolute(sobely)
    # Get the direction of the gradient
    grad = np.arctan2(sobely_abs, sobelx_abs)
    # Create a binary
    sobel_binary = np.zeros_like(gray_img)
    # Make 1's where the sobel_dir is within the thresholds
    sobel_binary[(grad >= thresh[0]) & (thresh[1] >= grad)] = 1

    return sobel_binary


def region_of_interest(img_undist, vertice):
    # defining a blank mask to start with
    mask = np.zeros_like(img_undist)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img_undist.shape) > 2:
        channel_count = img_undist.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertice, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img_undist, mask)

    return masked_image


def combine_thresholds(img_undist):
    # Region for the lane marking area:
    region = np.array([[(180, img_undist.shape[0]), (580, 450), (720, 450), (1200, img_undist.shape[0])]], dtype=np.int)

    result_x = abs_sobel_thresh(img_undist, 'x', sX_kernel, sX_thresh)
    result_y = abs_sobel_thresh(img_undist, 'y', sY_kernel, sY_thresh)
    result_mag = mag_thresh(img_undist, sMag_kernel, sMag_thresh)
    result_dir = dir_threshold(img_undist, sDir_kernel, sDir_thresh)
    result_hls = hls_Sspace(img_undist, hls_thresh)

    combined = np.zeros_like(result_x)
    combined[((((result_y == 1) & (result_dir == 1)) | ((result_dir == 1) & (result_mag == 1))) |
              ((result_dir == 1) & (result_hls == 1)))] = 1

    # ++ Old Combined ++
    # combined[(((result_x == 1) | (result_y == 1)) & ((result_dir == 1) | (result_hls == 1))) |
    #          ((result_hls == 1) & ((result_mag == 1) | (result_dir == 1)))] = 1
    combined_img = region_of_interest(combined, region)

    # ++ To check the output of the single thresholds ++
    # f, ([ax1, ax2], [ax3, ax4], [ax5, ax6]) = plt.subplots(3, 2, figsize=(15, 5))
    # ax1.imshow(result_x)
    # ax1.set_title('X-Sobel Image', fontsize=12)
    # ax2.imshow(result_y)
    # ax2.set_title('Y-Sobel Image', fontsize=12)
    # ax3.imshow(result_mag)
    # ax3.set_title('Magnitude Image', fontsize=12)
    # ax4.imshow(result_dir)
    # ax4.set_title('Direction Image', fontsize=12)
    # ax5.imshow(result_hls)
    # ax5.set_title('HLS Image', fontsize=12)
    # ax6.imshow(combined_img)
    # ax6.set_title('Combined Image', fontsize=12)
    # plt.show()

    return combined_img
