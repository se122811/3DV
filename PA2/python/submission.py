
import numpy as np 
import helper as hlp
import scipy.linalg
from scipy.spatial import distance
from scipy.ndimage import gaussian_filter
from scipy.linalg import rq


def eight_point(pts1, pts2, M):
    N = np.shape(pts1)[0]
    
    ones = np.ones((N,1)) #[NX1]
    pts1_h = np.concatenate((pts1, ones), axis=1) # [NX3]
    pts2_h = np.concatenate((pts2, ones), axis=1) 

    T = np.array([[1/M, 0, 0], [0, 1/M, 0], [0, 0, 1]]) #
    pts1_n = np.dot(T, pts1_h.T).T # [Nx3]
    pts2_n = np.dot(T, pts2_h.T).T # 

    A = np.zeros((N, 9))
    for m in range(N):
        xm,ym,_ = pts1_n[m,:] # (x_m,y_m,1)    (3,)
        xm_prime,ym_prime,_ = pts2_n[m,:] # (x_m`,y_m`,1)  (3,)

        Am = [xm*xm_prime, xm*ym_prime, xm, ym*xm_prime, ym*ym_prime, ym, xm_prime,ym_prime, 1]
        A[m,:] = Am

    u, s, vt = np.linalg.svd(A)

    F = vt[-1, :] 
    F = np.reshape(F, (3, 3))

    u, s, vt = np.linalg.svd(F)
    s[-1] = 0
    s_prime = np.diag(s)
    F = np.dot(np.dot(u, s_prime),vt)

    F = hlp.refineF(F, pts1/M, pts2/M) 

    F = np.dot(np.dot(T.T, F),T) 

    return F


def similarity(im1, im2, p1, pts2, w, pdist):
    d = w//2
    h2 = im2.shape[0]
    w2 = im2.shape[1]
    x1, y1 = p1 
    w1 = im1[y1-d:y1+d+1, x1-d:x1+d+1]

    dist = None
    final_match = None

    for j in range(pts2.shape[0]):
        p2 = pts2[j,:]
        x2, y2 = p2 
        if x2-d<0 or x2+d>=w2 or y2-d<0 or y2+d>=h2:
            pass
        elif np.linalg.norm(p1-p2) > pdist:
            pass
        else:
            win2 = im2[y2-d:y2+d+1, x2-d:x2+d+1]
            tmpDist = np.sum(np.square(gaussian_filter(w1 - win2, sigma=1.0)))
            if dist == None or tmpDist < dist:
                dist = tmpDist
                final_match = pts2[j,:]
    return final_match

def epipolar_correspondences(im1, im2, F, pts1):
    N = pts1.shape[0]
    ones = np.ones((N,1))
    pts1_h = np.concatenate((pts1, ones), axis=1)
    l2 = np.dot(F, pts1_h.T) # 3 x N
    wd = im2.shape[1] 
    x = np.array([np.arange(0, wd)]).T # x축 탐색 pixel 
    w = 19
    md = 40
    pts2 = np.zeros_like(pts1)
    for i in range(N): 
        # Scan the epipolar line to find the best match
        p1 = pts1[i,:] # (x1, y1)
        a, b, c = l2[:,i] # epipolar line elements
        y = np.round((- c - a * x) / b).astype(np.int) # ax+by+c=0 이용
        candidates = np.concatenate((x, y), axis=1)
        final_match = similarity(im1, im2, p1, candidates, w, md)
        pts2[i,:] = final_match
    return pts2

def essential_matrix(F, K1, K2):
    E = np.dot(K2.T, F) 
    E = np.dot(E, K1)
    return E


def triangulate(P1, pts1, P2, pts2):
    N = pts1.shape[0]
    P1_1 = P1[0,:] # 4, 
    P1_2 = P1[1,:]
    P1_3 = P1[2,:]
    P2_1 = P2[0,:]
    P2_2 = P2[1,:]
    P2_3 = P2[2,:]
    pts3d = np.zeros((N, 3))

    for i in range(N):
        x1, y1 = pts1[i,:]
        x2, y2 = pts2[i,:]
        first_1 = y1 * P1_3 - P1_2  # Pn_m is the m-th row of Pn
        second_1 = x1 * P1_3 - P1_1
        first_2 = y2 * P2_3 - P2_2 
        second_2 = x2 * P2_3 - P2_1

        A = np.vstack((first_1, second_1, first_2, second_2))

        u, s, vt = np.linalg.svd(A)

        X = vt[-1]
        X = X/X[-1]
        pts3d[i,:] = X[:3]
    return pts3d

def rectify_pair(K1, K2, R1, R2, t1, t2):
    c1 = -np.dot(np.linalg.inv(np.dot(K1, R1)), np.dot(K1, t1)) 
    c2 = -np.dot(np.linalg.inv(np.dot(K2, R2)), np.dot(K2, t2))

    r1 = (c1 - c2) / np.linalg.norm(c1 - c2)
    r1.transpose()[0]
    r2 = np.cross(r1.T[0], R1[2,:].T)
    r3 = np.cross(r2, r1.T[0])
    r1.transpose()[0]
    R_n = np.array([r1.transpose()[0], r2, r3])
    R1p = R_n 
    R2p = R_n
    
    K_n = K2 
    K1p = K_n
    K2p = K_n
    
    t1p = -np.dot(R_n, c1)
    t2p = -np.dot(R_n, c2)
    
    M1 = np.dot(np.dot(K1p, R1p), np.linalg.inv(np.dot(K1, R1)))
    M2 = np.dot(np.dot(K2p, R2p), np.linalg.inv(np.dot(K2, R2)))
    return M1, M2, K1p, K2p, R1p, R2p, t1p, t2p

def get_disparity(im1, im2, max_disp, win_size):
    r, c = im1.shape
    dispM = np.zeros_like(im1)
    w = (win_size-1)//2
    
    for i in range(max_disp+w+2, r-w-max_disp-2):
        for j in range(w+max_disp+2, c-w-2):
            distList = np.zeros(max_disp)
            for d in range(max_disp):
                win1 = im1[i-w:i+w+1, j-w:j+w+1]
                win2 = im2[i-w:i+w+1, j-w-d:j+w+1-d]
                tmpDist = np.linalg.norm(win1 + win2)
                distList[d] = tmpDist
            minIndex = np.argmin(distList)
            dispM[i,j] = minIndex

    return dispM


def get_depth(dispM, K1, K2, R1, R2, t1, t2):
    c1 = -np.dot(np.linalg.inv(np.dot(K1, R1)), np.dot(K1, t1)) # 3 x 1
    c2 = -np.dot(np.linalg.inv(np.dot(K2, R2)), np.dot(K2, t2))
    b = np.linalg.norm(c1 - c2)
    f = K1[0,0]
    with np.errstate(divide='ignore'):
        depthMap = np.divide(b*f, dispM)
    depthMap[depthMap == np.inf] = 0
    depthMap[depthMap == -np.inf] = 0
    return depthMap


def estimate_pose(x, X):
    n = x.shape[0]
    A = []
    for i in range(n):
        x1, y1, z1 = X[i, 0], X[i, 1], X[i, 2]
        u, v = x[i, 0], x[i, 1]
        A.append([x1, y1, z1, 1, 0, 0, 0, 0, -u*x1, -u*y1, -u*z1, -u])
        A.append([0, 0, 0, 0, x1, y1, z1, 1, -v*x1, -v*y1, -v*z1, -v])
    A = np.array(A)

    U, S, V = np.linalg.svd(A)

    L = V[-1, :]/V[-1, -1]
    P = L.reshape(3, 4)

    P = P * np.sign(np.linalg.det(P[:, :3]))
    return P

def estimate_params(P):
    U, S, V = np.linalg.svd(P[:, :3])
    c = (V[:, -1]/V[-1, -1]).T # center
    
    K, R = scipy.linalg.rq(P[:, :3], 3) # qr decomposition

    D = np.diag(np.sign(np.diag(K)))
    K = np.matmul(K, D)
    R = np.matmul(D, R)
    K = K/K[-1, -1]

    t = -np.dot(R, c) # t=-Rc
    return K, R, t
