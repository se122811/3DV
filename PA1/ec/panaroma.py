import numpy as np
import cv2

#Import necessary functions
import sys


#Write script for Q4.1
img_names = ['m./data/pano_left.jpg','m./data/pano_right.jpg']

imgs = []
for name in img_names:
    img = cv2.imread(name)
    
    if img is None:
        print('Image load failed!')
        sys.exit()
        
    imgs.append(img)

mode = cv2.STITCHER_SCANS
stitcher = cv2.Stitcher_create(mode)

status, dst = stitcher.stitch(imgs)
print(status)
print(dst)

if status != cv2.Stitcher_OK:
    print('Stitch failed!')
    sys.exit()
    
cv2.imwrite('../result/pano.jpg', dst)

# cv2.namedWindow('dst', cv2.WINDOW_NORMAL)
# cv2.imshow('dst',dst)
# cv2.waitKey()
# cv2.destroyAllWindows()


