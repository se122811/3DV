import numpy as np
import helper as hlp
import skimage.io as io
import submission as sub
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Load the two temple images and the points from data/some_corresp.npz (2.5-1)
I1 = io.imread('data/im1.png')
I2 = io.imread('data/im2.png')
data = np.load('data/some_corresp.npz')
pts1 = data["pts1"]
pts2 = data["pts2"]

# 2. Run eight_point to compute F  (2.5-2)
M_ = np.max([I1.shape[0], I1.shape[1]])
F = sub.eight_point(pts1, pts2, M_)
print(F)
#hlp.displayEpipolarF(I1, I2, F) # 2.1 in your write-up

# 3. Load points in image 1 from data/temple_coords.npz (2.5-3)
data1 = np.load('data/temple_coords.npz')
points1 = data1['pts1']

# 4. Run epipolar_correspondences to get points in image 2 (2.5-3)
points2 = sub.epipolar_correspondences(I1, I2, F, points1)
#hlp.epipolarMatchGUI(I1, I2, F) # 2.2 in your write-up

# (2.5-4)
cam = np.load('data/intrinsics.npz')
K1 = cam['K1']
K2 = cam['K2']
cam.close()
E = sub.essential_matrix(F, K1, K2)
# print(E)

# 5. Compute the camera projection matrix P1 (2.5-5)
I = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]]) # 3x4
P1 = K1.dot(I)
print(P1)

# 6. Use camera2 to get 4 camera projection matrices P2
M = hlp.camera2(E)
P_list = []
for i in range(4):
      P2 = np.dot(K2,M[:,:,i])
      P_list.append(P2)
P_l = np.array(P_list)

# 7. Run triangulate using the projection matrices
pt_list = []
for i in range(4):
      pt = sub.triangulate(P1, points1, P_l[i], points2)
      pt_list.append(pt)
pt_l = np.array(pt_list)

# 8. Figure out the correct P2
S_list = []
for i in range(4):
      s = np.sum(pt_l[i,:,-1]>=0)
      S_list.append(s)
S_l = np.array(S_list)

index = np.argmax(S_l)
P2 = P_l[index] # Correct P2 and 3d points
pt = pt_l[index]

# report the reprojection error
def reprojection_error(pt, pt3d, P):
    N = np.shape(pt)[0]
    pt3d_h = np.concatenate((pt3d, np.ones((pt3d.shape[0], 1))), axis=1)
    pt_p = np.dot(P, pt3d_h.T).T
    div = np.tile(pt_p[:,2],(2,1)).T
    pt_pp = pt_p[:,0:2]/div
    dist = np.sum(np.sqrt(np.sum((pt_pp-pt) ** 2, axis=1)))/N
    return dist


for i in range(4):
    e1 = reprojection_error(points1, pt_l[i], P1)
    e2 = reprojection_error(points2, pt_l[i], P_l[i])
    print('P2_{}'.format(i))
    print('{}\t{}'.format(e1,e2))

# 9. Scatter plot the correct 3D points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = pt[:,0]
y = pt[:,1]
z = pt[:,2]
ax.scatter(x, y, z, c='b', marker='o')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
print()

# 10. Save the computed extrinsic parameters (R1,R2,t1,t2) to data/extrinsics.npz
extrinsics = 'data/extrinsics.npz'
R1 = I[:,0:3]
t1 = np.array([I[:,3]]).T
M2 = np.dot(np.linalg.inv(K2), P2)
R2 = M2[:,0:3]
t2 = np.array([M2[:,3]]).T
np.savez(extrinsics, R1=R1, R2=R2, t1=t1, t2=t2)




