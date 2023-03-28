# Panoramas

## Synopsis

In this assignment, you will be writing code to align & stitch together a series of images into a panorama. You will then use your code on your own pictures to make a panorama. Take at least 3 images (although feel free to take more) to use for a panorama. You must take your own pictures for this assignment.

To help you with this assignment, you should carefully read the required reading from the Szeliski text , and watch the associated module:
  - Chapter 9.1: Szeliski, R. (2010). [Computer Vision: Algorithms and Applications](http://szeliski.org/Book/1stEdition.htm). Springer.
  - Module 05-03: Panorama


## Instructions

- Images in the `images/source/sample` directory are provided for testing -- *do not include these images in your submission* (although the output should appear in your report). 

- Downsampling your images to 1-2 MB each will save processing time during development. (Larger images take longer to process, and may cause problems for the autograder which is resource-limited.)

- We have provided a basic script to execute your panorama pipeline by running `python main.py`. The script will look inside each subfolder under `images/source` and attempt to apply the panorama procedure to the contents, and save the output to a folder with the same name as the input in `images/output/`. (For example, `images/source/sample` will produce output in `images/output/sample`.)


### 1. Implement the functions in the `panorama.py` file.

  - `getImageCorners`: Return the x, y coordinates for the four corners of an input image
  - `findHomography`: Return the transformation between the keypoints of two images
  - `getBoundingCorners`: Find the extent of a canvas large enough to hold two images
  - `warpCanvas`: Warp an input image to align with the next adjacent image
  - `createImageMask`: Create a mask representing valid portions of the image
  - `createRegionMasks`: Create three masks: one containing True for the post-warp(left) mask only, one that contains True for the region in which the right and left masks overlaps, and one containing True for the post-translated(right) mask only
  - `findDistanceToMask`: Return a distance map containing the distance from each True pixel to the nearest False pixel
  - `generateAlphaWeights`: Create a matrix of alpha weights based on the ratio of left and right distance maps
  - `blendImagePair`: Fit two images onto a single canvas

**Important Notes:** The `blendImagePair` function is NOT auto-scored. You will see the autograder return a 0/10 for blendImagePair(), which is EXPECTED and if you implemented the other required functions correctly, you will receive a score of 40/50 from the autograder. We will be manually grading blendImagePair(), so there is no need to make sure your blendImagePair() implementation passes the autograder. Submit to the autograder and receive your 40/50 first before moving on to blendImagePair(). Your blendImagePair() must be implemented using Python and you may include additional imports if necessary. 

In order to receive _any_ credit for the blendImagePair() function, you are required to replace the default insertion blend that we provide with your own blending function. Your resulting blend should match the quality of the default insertion blend provided in the images/example directory. You may use the functions `createImageMask, createRegionMasks, findDistanceToMask, generateAlphaWeights` to perform a blend, but this pipeline can be added to if desired. We want you to be creative. Good blending is difficult and time-consuming to achieve. We do not expect you to implement a universally perfect and seamless blend to get basic credit for the function. Make sure that your projection is correct-- the resulting image must be composted of the fully warped images from both images.  An incorrect (or "reversed") projection will "lop off" bits of one image or the other. They're supposed to overlap, but you're not supposed to be truncated in any way.  Truncated or cropped results will not receive credit.  The reason? There have been cases of students "hiding" bad results by manipulating the projection in such a way that the projected image is not fully represented.  You will lose far more points by obscuring results than by having defects.

A note about createRegionMasks.  Folks seem to have trouble with this one occasionally, generally by overcomplicating it.  Remember, the mask should represent the entire area of the projected region and have no gaps.  A lot of people use cv2.threshold for an unknown reason.  If you are tempted to do that here, you might want to rethink your strategy. In particular, folks get in trouble here because they forget that a positive fraction, no matter how small, is still greater than zero.  So many people get stuck on that.. It's weird, right?
   
The docstrings of each function contains detailed instructions. You are *strongly* encouraged to write your own unit tests based on the requirements. The `panorama_tests.py` file is provided to get you started. Your code will be evaluated on input and output type (e.g., uint8, float, etc.), array shape, and values. (Be careful regarding arithmetic overflow!)

When you are ready to submit your code, you can submit it to the Gradescope autograder for scoring, but we will enforce the following penalty for the submissions:
  * <= 20 submissions -> No penalty
  * <= 30 but >20 submissions -> -5 penalty
  * <= 40 but >40 submissions -> -10 penalty
  * more than 40 submissions  -> -20 penalty


### 2. Generate your own panorama

Once your code has passed the autograder and you’ve run the test inputs, you are ready to assemble your own panorama(s).  Choose a scene for your panorama and capture a sequence of at least three (3) partially overlapping frames spanning the scene. Your pictures should be clear (not be blurry) for feature matching to work well, and you should make sure you have substantial overlap between your images for best results. You will need to explore different feature matching and keypoint settings to improve your results, and to record your modifications and parameter settings in your report (see the report template).

### 3. Complete and Submit the Report on Gradescope AND Canvas

- The total size of your report+resources must be less than **20MB** for this project. If your submission is too large, you can reduce the scale of your images or report. You can compress your report using [Smallpdf](https://smallpdf.com/compress-pdf) or other apps.

Download a copy of the A3 report template provided in this directory and answer all of the questions. You may add additional slides to your report if necessary. Save your report as `report.pdf`. Submit your PDF to **A3-Panoramas Report** on Gradescope. 

After you upload your PDF, you will be taken to the "Assign Questions and Pages" section that has a Question Outline on the left hand side. These outline items are determined by the Instructors. **For each question - select the question, and then select ALL pages that go with that question. This is important to do correctly because any pages that are not selected for the corresponding question might get missed during grading.**

In order for your report to be uploaded to the Peer Feedback system, submit your PDF on **Canvas > Assignments > A3:Panoramas Report**. This is required. 


### 4. Submit the Code and Images

Create an archive named `resources.zip` containing your input images and final artifact. Your images must be one of the following types: jpg, jpeg, bmp, png, tif, or tiff.

Your files in `resources.zip` must be named as follows:
  * `panorama.py` - python file
  * `main.py` - your python file you wrote to generate the final results. You can modify this, it won't be graded.
  * `input_1.jpg` - first image to be stiched together to create a panorama
  * `input_2.jpg` - second image to be stiched together to create a panorama
  * `input_3.jpg` - third image to be stiched together to create a panorama
  * `result.jpg` - final panorama
  * Additional `input_*.jpg` as desired, but input_1, input_2 and input_3 are required.

When you are zipping up your files, make sure that you zip only the files, and not the folder containing them. Gradescope looks at the root of the zip archive for the submitted files, so keep this in mind to avoid submission issues. Submit your zip to **A3-Panoramas Resources** on Gradescope. 

The autograder will test `panorama.py` and it will also check that you submitted the four image files with the correct names and acceptable file extensions (e.g. `input_1.jpg`, `result.jpg`). For the images, the autograder only checks that you submitted the required image files with the correct name and an acceptable extension. The Instructors will manually check all resources for correctness/quality during grading and will deduct points as necessary. Your code must run in less than 10 minutes on the autograder.
      
**Notes:**

  - Your final Resources submission must include all of the files listed in section 4. Gradescope does not keep track of files submitted earlier, and if your submission is incomplete, you cannot add any additional files after the due dates. 
  
  - **EXIF metadata:** When including or sharing images, make sure there are no data contained in the EXIF metadata that you do not want shared (i.e. GPS). If there is, make sure you strip it out before submitting your work or sharing your photos with others. Normally, we require that your images include aperture, shutter speed, and ISO. You may use `pypi/exif` library in a separate python file to strip GPS data if you wish.  Make sure you do not use an app that strips all exif from the image. 

  - **DO NOT USE 7zip.** We've had problems in the past with 7z archives, so please don't use them unless you don't mind getting a zero on the assignment.
  
  - **Note for Mac users:** If you zip your files, do it from within the folder by selecting the files.  Do not zip the directory.  What happens is that the Mac adds extra stuff and that might confuse Gradescope.


## Criteria for Evaluation

Your submission will be graded based on:

  - Correctness of required code.
  - Overall quality of results.  Creativity is encouraged, but not required. Check out the comments in the LaTex doc.
    Everybody likes being creative though, right?
  - Completeness and quality of report.


## Assignment FAQ

- Can we crop the resulting panorama in our report?

  Yes, you may include a cropped panorama in the report, but you **MUST** also include the original **uncropped** version in your report. The functions we ask you to fill in expect uncropped image inputs and outputs. As mentioned above, the final result must represent the full projection of both sets of images.  If one image is projected into a smaller rectangle, that will result in zero points for that section and maybe some eyeball rolling on my part.  

- Can we add intermediate functions to our code for the blending portion of the assignment?

  Yes, but it is your responsibility to make sure your code passes all tests on the autograder.

- Can I use a tool like Photoshop to improve the blending result? 

  No, you may not use any other software; you must only use your own code to generate a final panorama. You are only allowed to use a simple image editing tool for cropping the final result (if you so choose), but don't forget to include the **uncropped** originals.

- Can I add extra python code files in my submission? 

  No, all code you write for this assignment must be contained in the panorama.py file and main.py
