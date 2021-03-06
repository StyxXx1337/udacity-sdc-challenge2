## Advanced Lane Lines Project

### Second Project in the Udacity Self-Driving Engineer Course

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undist_image.png "Undistorted"
[image2]: ./test_images/test4.jpg "Road Transformed"
[image3]: ./output_images/combined_image.jpg "Binary Example"
[image4]: ./output_images/warped_image.jpg "Warp Example"
[image5]: ./output_images/lines_image.png "Fit Visual"
[image6]: ./output_images/final_image.png "Output"
[video1]: ./project_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the `CalibrateCamera.py` from line 6 to 41 located in "./pyModules/CalibrateCamera.py".  

I start by preparing the basic data to start extracting the "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. The chessboard which is used in the calibration has 9 corners in x-Direction and 6 corners in y-Direction. 

Further I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `obj_p` is just a replicated array of coordinates, and `obj_pts` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `img_pts` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  


I then used the output `obj_pts` and `img_pts` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. 
Additional I created function `undistort_image` in the same file from line 48 to 50 to undistort the image based on the output from the camera calibration using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

Undistorted Image:
![alt text][image1]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps are in `ColorSpace.py`).  Here's an example of my output for this step.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `get_matrixes()`, which appears in lines 5 through 24 in the file `PerstepctiveTransform.py` (pyModules/PerstepctiveTransform.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), and has source (`src`) and destination (`dst`) points hardcoded in the following manner:

```python
    # Assumption that the camera is in the middle of the car
    src = np.float32(
        [[(img_size[0] // 2) - 500, img_size[1]],
         [((img_size[0] // 2) - 100), 480],
         [((img_size[0] // 2) + 100), 480],
         [(img_size[0] // 2) + 500, img_size[1]]])

    # Destination Points of the image
    dst = np.float32(
        [[200, img_size[1]],
         [200, 0],
         [img_size[0] - 200, 0],
         [img_size[0] - 200, img_size[1]]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 140, 720      | 200, 720      | 
| 540, 480      | 200, 0        |
| 740, 480      | 1040, 0       |
| 1140, 720     | 1040, 720     |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in the function `measure_curvature_real()` lines 24 through 35 in my code in `Utilities.py` (pyModules/Utilities.py).

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in 2 functions. I created `visualize_lane()` to visualize the lines from 62 through 83 in my code in `Utilities.py`. 
Further I implemented the function `print_on_image()` to print the results of the Offset and Radius calculation on the picture. Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

   1. In the test video in at time 0:41 the differences in light made it very difficult to keep a good track of the lane.
      So more challenging lighting like for example driving through forest with many changes in light.
   2. Many curving situations would also cause a lot of challenges for that algorithm, since the curve radius change very fast.
