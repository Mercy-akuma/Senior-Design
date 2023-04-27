import cv2
import os
import numpy as np

os.chdir(os.path.dirname(__file__))
# Load the optical and thermal images
optical_img = cv2.imread('Figure/test_rgb_image.jpg')
# Load the thermal image
thermal_img = cv2.imread('Figure/test_thermal.png')

# Initiate SIFT detector
sift = cv2.SIFT_create()

# Find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(optical_img,None)
kp2, des2 = sift.detectAndCompute(thermal_img,None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

# Perform perspective transformation based on the keypoints
src_pts = np.float32([ kp1[m[0].queryIdx].pt for m in good ]).reshape(-1,1,2)
dst_pts = np.float32([ kp2[m[0].trainIdx].pt for m in good ]).reshape(-1,1,2)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

# Check if the transformation matrix is valid
if M is None or M.size == 0:
    print("Error: Transformation matrix is invalid")
else:
    # Apply the transformation to the optical image
    h,w = optical_img.shape[:2]
    transformed_img = cv2.warpPerspective(optical_img, M, (w, h))

    # Save the output image
    cv2.imwrite('Figure/registered_image.jpg', transformed_img)
























