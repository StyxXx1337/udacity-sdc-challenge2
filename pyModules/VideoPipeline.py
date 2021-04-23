# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def process_image(image):
    ## Get Image and Undistrot Image
    #img = np.copy(image)
    img_undist = undistort_image(image, mtx_camera, dist_camera)

    ## Create a binary Image
    result_x = abs_sobel_thresh(img_undist, 'x', sX_kernel, sX_thresh)
    result_y = abs_sobel_thresh(img_undist, 'y', sY_kernel, sY_thresh)
    result_mag = mag_thresh(img_undist, sMag_kernel, sMag_thresh)
    result_dir = dir_threshold(img_undist, sDir_kernel, sDir_thresh)
    result_hls = hls_Sspace(img_undist, hls_thresh)

    combined = np.zeros_like(result_x)
    combined[((result_x == 1) & (result_y == 1)) | (((result_mag == 1) & (result_dir == 1))) | (result_hls == 1)] = 1
    combined_img = region_of_interest(combined, region)

    ### Create Image Size Parameters for warpPerspective
    mtx_persp, mtx_persp_inv = get_matrixes()
    warped = warp_picture(combined_img, mtx_persp)

    ## Detect the Lines in the warped Image
    ### Get the Polynomiyal for the left and right side
    ### also get the amount of pixels detected to use later as a weighting.
    ### The more pixles detected the higher the accuracy.
    out_img, left_poly, right_poly, poly, leftx, rightx = fit_polynomial(warped)

    ## Getting Curvature for left and right side
    left_r, right_r = measure_curvature_real(poly, left_poly, right_poly, xm_per_pix, ym_per_pix)

    ## Weight the curvature ratios based on the amount of pixles
    weighted_r = (((left_r * leftx.shape[0]) + (right_r * rightx.shape[0])) / (((leftx.shape[0] + rightx.shape[0]))))
    weighted_r = math.trunc(weighted_r * 100) / 100
    radius_str = "The Curve radius is: " + str(weighted_r) + 'm'

    ## Get the position and offset in the lane
    position = (np.average(leftx) + np.average(rightx)) / 2
    offset = ((img_size[0]/2)-position)*xm_per_pix
    offset = math.trunc(offset * 100) / 100
    offset_str = "Offset: " + str(offset) + 'm'

    ## Print the lines back into the image:
    font = cv2.FONT_HERSHEY_SIMPLEX
    mtx_persp_inv = cv2.getPerspectiveTransform(dst, src)
    lines_img = cv2.warpPerspective(out_img, mtx_persp_inv, img_size, flags=cv2.INTER_LINEAR)
    output_img = cv2.addWeighted(img_undist[...,::-1], .8, lines_img, 0.9, 0.2)
    cv2.putText(output_img,radius_str,(10,50), font, 1.2,(255,255,255),2, cv2.LINE_AA)
    cv2.putText(output_img,offset_str,(10,100), font, 1.2,(255,255,255),2, cv2.LINE_AA)

    return output_img


# In[13]:


mtx_camera, dist_camera = calibrate_camera()
white_output = 'project_video.mp4'
clip1 = VideoFileClip("project_video.mp4").subclip(0,5)
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
get_ipython().run_line_magic('time', 'white_clip.write_videofile(white_output, audio=False)')
