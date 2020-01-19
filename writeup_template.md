
# **Advanced Lane Finding Project** 

---

Main goal: Finding Lanes.

Challenges: 
* Images from the camera are distorted
* Images taken at bad lighting conditions
* Difficult to detect lanes because of the lane color

---
The above mentioned challenges make it harder to find the lanes from the camera images. Camera images are processed in the following manner to find the lanes from the raw camera images.

* **Camera calibration:** Compute the camera calibration matrix and distortion coefficients given a set of chessboard images and apply the calculated distortion correction to the raw images.
* **Gradients and color transforms:** Use color transforms, gradients, etc., to create a thresholded binary image.
* **Perspective Transformation:** Apply a perspective transform to rectify binary image ("birds-eye view").
* **Lane Detection:** Detect lane pixels and fit to find the lane boundary.
* **Estimating the curvature:** Determine the curvature of the lane and vehicle position with respect to center.
* **Visualizing lanes and calculated curvature:** Warp the detected lane boundaries back onto the original image.


[//]: # (Image References)

[undist_checker_board]: ./output_images/undistorted/Checker_board.jpg "Undistorted Checker Board"
[undist_test_img]: ./output_images/undistorted/test_img.jpg "Undistorted Test image"
[c_spaces_img1]: ./output_images/color_spaces/img1.jpg "Test image"
[c_spaces_img1_sobel]: ./output_images/color_spaces/img1_sobel.jpg "Sobel-x"
[c_spaces_img1_s_space]: ./output_images/color_spaces/img1_s_space.jpg "S space from HLS"
[c_spaces_img1_combined]: ./output_images/color_spaces/img1_combined.jpg "Sobel and S combined"
[c_spaces_img2]: ./output_images/color_spaces/img2.jpg "Test image"
[c_spaces_img2_sobel]: ./output_images/color_spaces/img2_sobel.jpg "Sobel-x"
[c_spaces_img2_s_space]: ./output_images/color_spaces/img2_s_space.jpg "S space from HLS"
[c_spaces_img2_combined]: ./output_images/color_spaces/img2_combined.jpg "Sobel and S combined"
[perspective_transform]: ./output_images/perspective_transform/perspective_transform.png "Perspective Transformation"
[lanes]: ./output_images/lane_detection/lane_detection.jpg "Lane Detection"
[video1]: ./project_video.mp4 "Video"

---
## Pipeline

#### **1. Camera calibration:**
The images from the cameras are generally distorted either because of the lens defects or the improper alignment of the camera.

In this step, distortion coefficients of the camera are calculated. This is estimated by taking multiple images of a known checker board pattern aligned in various positions.

Corners in these images are obtained from the OpenCv function 'cv2.findChessboardCorners'. The actual corners of the chessboard grid (called as `objpoints`) are associated to the corners in the images from the camera (called as `imgpoints`). The distortion coefficients are calculated by feeding the arrays  `objpoints` and `imgpoints` to the function `cv2.calibrateCamera()`. The correction is applied to the images using the `cv2.undistort()` function.


![alt text][undist_checker_board]

![alt text][undist_test_img]


#### **2. Gradients and color transforms:** 

#### (i) Sobel Operator

#### (ii) Color spaces

2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

| Image| Sobel Transform| S space| Combined|
|:-:|:-:|:-:|:-:|
|![alt text][c_spaces_img1]|![alt text][c_spaces_img1_sobel]|![alt text][c_spaces_img1_s_space]|![alt text][c_spaces_img1_combined]|
|![alt text][c_spaces_img2]|![alt text][c_spaces_img2_sobel]|![alt text][c_spaces_img2_s_space]|![alt text][c_spaces_img2_combined]|

#### **3. Perspective Transformation:**
 Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:


following hardcoded source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| [260, 655 ]   | [280, 655 ]   | 
| [560, 470 ]   | [280, 0   ]   |
| [730, 470 ]   | [1020, 0  ]   |
| [1040, 655]   | [1020, 655]   |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][perspective_transform]

#### **4. Lane Detection:**
 Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][lanes]

#### **5. Estimating the curvature:**
5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### **6. Visualizing lanes and calculated curvature:**
6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
