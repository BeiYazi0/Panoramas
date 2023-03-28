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
