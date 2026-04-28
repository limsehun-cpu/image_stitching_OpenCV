import numpy as np
import cv2 as cv

# Image blending
def blending(copy, before_copy, size):
    alpha = np.linspace(0, 1, 30).reshape(1, -1, 1)
    copy[:,size-30:size] = (alpha * before_copy[:,size-30:size] + (1 - alpha) * copy[:,size-30:size]).astype(np.uint8)
    
    return copy

def stitching(img1, img2):
    # Retrieve matching points
    brisk = cv.BRISK.create()
    keypoints1, descriptors1 = brisk.detectAndCompute(img1, None)
    keypoints2, descriptors2 = brisk.detectAndCompute(img2, None)

    fmatcher = cv.DescriptorMatcher.create('BruteForce-Hamming')
    match = fmatcher.match(descriptors1, descriptors2)

    # Calculate planar homography and merge them
    pts1, pts2 = [], []
    for i in range(len(match)):
        pts1.append(keypoints1[match[i].queryIdx].pt)
        pts2.append(keypoints2[match[i].trainIdx].pt)
    pts1 = np.array(pts1, dtype=np.float32)
    pts2 = np.array(pts2, dtype=np.float32)

    H, inlier_mask = cv.findHomography(pts2, pts1, cv.RANSAC)
    img_merged = cv.warpPerspective(img2, H, (img1.shape[1] * 2, img1.shape[0]))
    before_copy = img_merged.copy()
    img_merged[:,:img1.shape[1]] = img1 # Copy

    return blending(img_merged, before_copy, img1.shape[1])

# Load images
img1 = cv.imread('image1.jpg')
img2 = cv.imread('image2.jpg')
img3 = cv.imread('image3.jpg')
assert (img1 is not None) and (img2 is not None) and (img3 is not None), 'Cannot read the given images'

merge = stitching(img2, img3)
merge = stitching(img1, merge)

# Show the merged image
cv.imshow('Image Stitching', merge)
cv.waitKey(0)
cv.destroyAllWindows()