**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[video1]: ./output_project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I adapted the code from lesson 35, HOG Sub-Sampling Window Search, in function
`find_cars` on line 15 of `hog_subsample.py`.  The function works in 2 modes:
training and predicting.  When training, the whole image is used and there is
only 1 window per image.

####2. Explain how you settled on your final choice of HOG parameters.

I used the default parameters from the lesson.  They turned out to work pretty
well after applying a multi-frame heat map and threshold so I didn't do any
parameter tuning.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the provided PNG images starting on line 115.  The training is saved for reuse.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used the HOG sub-sampling window search on line 52 of `hog_subsample.py`.
Therefore, HOG was only used one once per frame and then scaled windows were
searched with color features.  I used 5 different scales, 1, 1.5, 2, 2.5, and 3
over a region in about the bottom third of the frame through trial and error.

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YCrCb 3-channel HOG features plus
spatially binned color and histograms of color in the feature vector, which
provided a nice result.

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video,
function `process_image` on line 165 of `hog_subsample.py`.  From the positive
detections I created a heatmap and then thresholded that map to identify
vehicle positions.  I then used `scipy.ndimage.measurements.label()` to
identify individual blobs in the heatmap.  I then assumed each blob
corresponded to a vehicle.  I constructed bounding boxes to cover the area of
each blob detected.  

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The results turned out pretty well.  However, the speed is pretty terrible
taking ~30 minutes for the 50s project video.

