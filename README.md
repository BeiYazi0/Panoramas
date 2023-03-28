# Panoramas

## Instructions

- We have provided a basic script to execute your panorama pipeline by running `python main.py`. The script will look inside each subfolder under `images/source` and attempt to apply the panorama procedure to the contents, and save the output to a folder with the same name as the input in `images/output/`. (For example, `images/source/sample` will produce output in `images/output/sample`.)


### Implement the functions in the `panorama.py` file.

  - `getImageCorners`: Return the x, y coordinates for the four corners of an input image
  - `findHomography`: Return the transformation between the keypoints of two images
  - `getBoundingCorners`: Find the extent of a canvas large enough to hold two images
  - `warpCanvas`: Warp an input image to align with the next adjacent image
  - `createImageMask`: Create a mask representing valid portions of the image
  - `createRegionMasks`: Create three masks: one containing True for the post-warp(left) mask only, one that contains True for the region in which the right and left masks overlaps, and one containing True for the post-translated(right) mask only
  - `findDistanceToMask`: Return a distance map containing the distance from each True pixel to the nearest False pixel
  - `generateAlphaWeights`: Create a matrix of alpha weights based on the ratio of left and right distance maps
  - `blendImagePair`: Fit two images onto a single canvas

In order to receive _any_ credit for the blendImagePair() function, you are required to replace the default insertion blend that we provide with your own blending function. Your resulting blend should match the quality of the default insertion blend provided in the images/example directory. You may use the functions `createImageMask, createRegionMasks, findDistanceToMask, generateAlphaWeights` to perform a blend, but this pipeline can be added to if desired. We want you to be creative. Good blending is difficult and time-consuming to achieve. We do not expect you to implement a universally perfect and seamless blend to get basic credit for the function. Make sure that your projection is correct-- the resulting image must be composted of the fully warped images from both images.  An incorrect (or "reversed") projection will "lop off" bits of one image or the other. They're supposed to overlap, but you're not supposed to be truncated in any way.  Truncated or cropped results will not receive credit.  The reason? There have been cases of students "hiding" bad results by manipulating the projection in such a way that the projected image is not fully represented.  You will lose far more points by obscuring results than by having defects.

A note about createRegionMasks.  Folks seem to have trouble with this one occasionally, generally by overcomplicating it.  Remember, the mask should represent the entire area of the projected region and have no gaps.  A lot of people use cv2.threshold for an unknown reason.  If you are tempted to do that here, you might want to rethink your strategy. In particular, folks get in trouble here because they forget that a positive fraction, no matter how small, is still greater than zero.  So many people get stuck on that.. It's weird, right?
