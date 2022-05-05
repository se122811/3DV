import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection

import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection


def matchPics(I1, I2):
	#I1, I2 : Images to match

	#Convert Images to GrayScale
	gray1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
	gray2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)
	
	#Detect Features in Both Images
	locs_1 = corner_detection(gray1)
	locs_2 = corner_detection(gray2)

	#Obtain descriptors for the computed feature locations
	desc1, locs1 = computeBrief(gray1, locs_1)
	desc2, locs2 = computeBrief(gray2, locs_2)
	
	#Match features using the descriptors
	matches = briefMatch(desc1, desc2)
	
	return matches, locs1, locs2