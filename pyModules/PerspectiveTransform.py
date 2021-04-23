import cv2
import numpy as np

def get_matrixes(img_size):
    ## For the Image Transformation
    ### Source Points of the image
    ### Assumption that the camera is in the middle of the car
    src = np.float32(
            [[(img_size[0]//2)-500, img_size[1]],
            [((img_size[0]//2)-100), 480],
            [((img_size[0]//2)+100), 480],
            [(img_size[0]//2)+500, img_size[1]]])

    ### Destination Points of the image
    dst = np.float32(
            [[200,img_size[1]],
            [200, 0],
            [img_size[0]-200, 0],
            [img_size[0]-200, img_size[1]]])

    mtx_persp = cv2.getPerspectiveTransform(src, dst)
    mtx_persp_inv = cv2.getPerspectiveTransform(dst, src)
    return (mtx_persp, mtx_persp_inv)


def warp_picture(combined_img, mtx_persp):

    img_size = (combined_img.shape[1], combined_img.shape[0])
    warped_img = cv2.warpPerspective(combined_img, mtx_persp, img_size, flags=cv2.INTER_LINEAR)

    return warped_img
