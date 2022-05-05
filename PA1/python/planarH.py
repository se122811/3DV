import numpy as np
import cv2
from matchPics import matchPics
import random


def computeH(x1, x2):
    #Q2.2.1
    # #Compute the homography between two sets of points
    pt1 = x1
    pt2 = x2
    A = []
    for i in range(0, len(pt1)):
        x, y = pt1[i][0], pt1[i][1]
        x_p, y_p = pt2[i][0], pt2[i][1]
        A.append([x_p, y_p, 1, 0, 0, 0, -x*x_p, -y_p*x, -x])
        A.append([0, 0, 0, x_p, y_p, 1, -x_p*y, -y*y_p, -y])
        # A.append([x, y, 1, 0, 0, 0, -x*x_p, -y*x_p, -x_p])
        # A.append([0, 0, 0, x, y, 1, -x*y_p, -y*y_p, -y_p])
    
    A = np.asarray(A)
    U, S, Vh = np.linalg.svd(A)
    L = Vh[-1,:] / Vh[-1,-1]
    H2to1 = L.reshape(3, 3)
    
    return H2to1


def computeH_norm(x1, x2):
    #Q2.2.2
    #Compute the centroid of the points
    centroid_x1 = np.mean(x1, axis=0)
    centroid_x2 = np.mean(x2, axis=0)
    
    #Shift the origin of the points to the centroid
    x1_hc = np.hstack((x1, [[1], [1], [1], [1]])) # Homogeneous coordinates
    x2_hc = np.hstack((x2, [[1], [1], [1], [1]])) # Homogeneous coordinates
    
    T_x1 = [
            [1, 0, -centroid_x1[0]],
            [0, 1, -centroid_x1[1]],
            [0, 0, 1],
           ]
    T_x2 = [
            [1, 0, -centroid_x2[0]],
            [0, 1, -centroid_x2[1]],
            [0, 0, 1],
           ]
           
    x1_shifted = np.dot(T_x1, x1_hc.T)
    x1_shifted = x1_shifted.T
    x2_shifted = np.dot(T_x2, x2_hc.T)
    x2_shifted = x2_shifted.T
    
    #Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    dist_from_orig_x1 = np.linalg.norm(x1_shifted[:, :2], axis=1)
    max_dist_from_orig_x1 = np.max(dist_from_orig_x1)
    S_x1 = [
            [np.sqrt(2)/max_dist_from_orig_x1, 0, 0],
            [0, np.sqrt(2)/max_dist_from_orig_x1, 0],
            [0, 0, 1],
           ]
    dist_from_orig_x2 = np.linalg.norm(x2_shifted[:, :2], axis=1)
    max_dist_from_orig_x2 = np.max(dist_from_orig_x2)
    S_x2 = [
            [np.sqrt(2)/max_dist_from_orig_x2, 0, 0],
            [0, np.sqrt(2)/max_dist_from_orig_x2, 0],
            [0, 0, 1],
           ]
    x1_shifted_scaled = np.dot(S_x1, x1_shifted.T)
    x1_shifted_scaled = x1_shifted_scaled.T
    x2_shifted_scaled = np.dot(S_x2, x2_shifted.T)
    x2_shifted_scaled = x2_shifted_scaled.T
    
    #Similarity transform 1
    Sm_x1 = np.dot(S_x1, T_x1)
    
    #Similarity transform 2
    Sm_x2 = np.dot(S_x2, T_x2)
    
    #Compute homography
    H2to1 = computeH(x1_shifted_scaled, x2_shifted_scaled)
    #Denormalization
    H2to1 = np.dot(np.linalg.inv(Sm_x1), np.dot(H2to1, Sm_x2))
    H2to1 /= H2to1[2, 2]

    # x1_estimated = np.dot(np.linalg.inv(Sm_x1), np.dot(H2to1, np.dot(Sm_x2, x2_hc.T)))
    # x1_estimated = x1_estimated.T
    x1_estimated = np.dot(H2to1, x2_hc.T)
    x1_estimated = x1_estimated.T
    x1_estimated[:, :2]/x1_estimated[:, 2:3]-x1
    
    return H2to1


def computeH_ransac(locs1, locs2):
    #Q2.2.3
    # #Compute the best fitting homography given a list of matching points
    
    locs1 = np.array(locs1)
    locs2 = np.array(locs2)
    num = len(locs1)
    pt1 = np.zeros((4,2))
    pt2 = np.zeros((4,2))
    inliers = 0

    pt1 = locs1[0:4,:]
    pt2 = locs2[0:4,:]
    
    bestH2to1 = computeH_norm(pt1, pt2)
    # bestH2to1 = computeH(pt1, pt2)

    for i in range(2000):
        inliers_num = 0
        rand= random.sample(range(0, num), 4)
        pt1 = locs1[rand,:]
        pt2 = locs2[rand,:]
        H = computeH_norm(pt1, pt2)
        # H = computeH(pt1, pt2)
        H = np.asmatrix(H)

        for i in range(num):
            x = np.array([locs2[i,0], locs2[i,1], 1])
            x = np.asmatrix(x)
            # x_p = x * H
            x_p =  H * x.T
            x_p = x_p / x_p[2,0]

            if  x_p.A[0,0]- 2 < locs1[i,0] and  locs1[i,0] < x_p.A[0,0]+2 :
                if x_p.A[1,0]- 2 < locs1[i,1] and  locs1[i,1] < x_p.A[1,0]+2 :
                    inliers_num += 1
                    
        if inliers < inliers_num :
            bestH2to1 = H.A
            inliers = inliers_num
            
    print('inliers : ', inliers)
    
    return bestH2to1, inliers


def compositeH(H2to1, template, img):
    #Create a composite image after warping the template image on top
    #of the image using the homography
    #Note that the homography we compute is from the image to the template;
    #x_template = H2to1*x_photo
    #For warping the template to the image, we need to invert it.
    H2to1=np.linalg.inv(H2to1)
    
    height, width, channel = template.shape
    img_heigth, img_width, img_channel = img.shape

    #Create mask of same size as template
    mask = np.ones((height, width), dtype="uint8")
    
    # # #Warp mask by appropriate homography
    warped_mask = cv2.warpPerspective(mask, H2to1,(img_width, img_heigth))
    
    
    for i in range(img_heigth):
        for j in range(img_width):
            if warped_mask[i,j] == 0:
                warped_mask[i,j] = 1
            else :
                warped_mask[i,j] = 0

    # #Warp template by appropriate homography
    warped_templete = cv2.warpPerspective(template, H2to1, (img_width,img_heigth))
    
    # Use mask to combine the warped template and the image
    warped_mask =np.reshape(warped_mask,(warped_mask.shape[0], warped_mask.shape[1],1))
    img1 = warped_mask * img
    
    composite_img = img1 + warped_templete

    return  composite_img