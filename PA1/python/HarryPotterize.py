import numpy as np
import cv2
import skimage.io 
import skimage.color
#Import necessary functions
from planarH import computeH_ransac, compositeH
from matchPics import matchPics


# Write script for Q2.2.4


# 1. Reads cv.jpg, cv_desk.png, and hp_cover.jpg.
cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')
hp_cover = cv2.imread('../data/hp_cover.jpg')


# 2. Computes a homography automatically using MatchPics and computeH_ransac.
matches, locs1, locs2 = matchPics(cv_cover,cv_desk)

pt1 = np.zeros((len(matches), 2))
pt2 = np.zeros((len(matches), 2))

pt1_new = np.zeros((len(matches), 2))
pt2_new = np.zeros((len(matches), 2))

for i in range(len(matches)): 
    pt1[i,:] = locs1[matches[i][0]]
    pt2[i,:] = locs2[matches[i][1]]
    
pt1_new[:,0] = pt1[:,1]
pt1_new[:,1] = pt1[:,0]

pt2_new[:,0] = pt2[:,1]
pt2_new[:,1] = pt2[:,0]

# H, status = cv2.findHomography(pt1_new, pt2_new, cv2.RANSAC, 5.0)
H_2,l = computeH_ransac(pt1_new, pt2_new)
# H_2 = np.linalg.inv(H_2)

# 3. Uses the computed homography to warp hp_cover.jpg to the dimensions of the cv_desk.png image using the
# skimage function skimage.transform.warp or OpenCV function cv2.warpPerspective.
composite_img = compositeH(H_2, hp_cover, cv_desk)
cv2.imwrite("../data/b.jpg", composite_img)



# 4. At this point you should notice that although the image is being warped to the correct location, it is not filling up the
# same space as the book. Why do you think this is happening? How would you modify hp_cover.jpg to fix this issue?

'''
cv_cover and hp_cover have different image size. 
We can resize of hp_cover image.

'''

# 5. Implement the function:

resize_img = cv2.resize(hp_cover, dsize = (cv_cover.shape[1], cv_cover.shape[0]),  interpolation=cv2.INTER_AREA)
composite_img = compositeH(H_2, resize_img, cv_desk)

cv2.imwrite("../data/b_resize.jpg", composite_img)