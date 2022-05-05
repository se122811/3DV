import numpy as np
import cv2
import skimage.io 
import skimage.color
#Import necessary functions
from planarH import computeH_ransac, compositeH
from matchPics import matchPics
from loadVid import loadVid

#Write script for Q3.1


# 1. Reads cv_couver.jpg
cv_cover = cv2.imread('../data/cv_cover.jpg')

# 2. Reads book.mov, ar_source.mov
cv_desk = loadVid('../data/book.mov')
cv_desk_frames = cv_desk.shape[0]

ar_souce = loadVid('../data/ar_source.mov')
ar_source_frames = ar_souce.shape[0]

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('../result/ar_result.avi',fourcc, 15, (cv_desk.shape[2],cv_desk.shape[1]))


for f in range(ar_source_frames):
# for f in range(100):
    matches, locs1, locs2 = matchPics(cv_cover,cv_desk[f])

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
    if f % 5 == 0:
        H_2,l = computeH_ransac(pt1_new, pt2_new)
        # H_2 = np.linalg.inv(H_2)


    # 5. Implement the function:
    resize_img = cv2.resize(ar_souce[f], dsize = (cv_cover.shape[1], cv_cover.shape[0]),  interpolation=cv2.INTER_AREA)
    composite_img = compositeH(H_2, resize_img, cv_desk[f])
    out.write(composite_img)

out.release()