from matplotlib import widgets
import numpy as np
import cv2
from matchPics import matchPics
from scipy import ndimage
import matplotlib.pyplot as plt
from helper import plotMatches
# Q2.1.5
#Read the image and convert to grayscale, if necessary
cv_cover = cv2.imread('../data/cv_cover.jpg')
hist_matchs = np.zeros(36)
X = np.arange(0, 360, 10)

for i in range(36):
	#Rotate Image
	img_rotate = ndimage.rotate(cv_cover, 10*i, reshape=False)
	
	#Compute features, descriptors and Match features
	matches, locs1, locs2 = matchPics(cv_cover, img_rotate)
	counts = len(matches)
	if i==0 or i==9 or i==18:
		plotMatches(cv_cover, img_rotate, matches, locs1, locs2)
	#Update histogram
	hist_matchs[i] = counts
	
# Display histogram
hist = plt.hist(X, bins = 36, weights = hist_matchs)
plt.xlabel('Degree')
plt.ylabel('Matching keypoints')
plt.show()
