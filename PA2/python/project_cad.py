# Q4.3
import numpy as np
import submission as sub
import cv2 
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Circle
import tqdm

## 1. Load files
pnp_path = 'data/pnp.npz'
pnp = np.load(pnp_path, allow_pickle=True)

xyz = pnp['X']
uv = pnp['x']
img = pnp['image']
cad_pts = pnp['cad'][0, 0][0]
cad_line = pnp['cad'][0, 0][1]-1

## 2. Run estimate_pose & estimate_paramas 
P = sub.estimate_pose(uv, xyz)
K, R, t = sub.estimate_params(P)

## 3. Project the given 3D points X onto the image ( using camera matrix P )
w = np.ones((xyz.shape[0], 1))
xyz = np.concatenate((xyz, w), axis=1)

new = np.matmul(P, xyz.T)
new = new/new[-1]
new = np.transpose(new)

fig,ax = plt.subplots(1)
for i, j, _ in new:
    circ = Circle((i, j), 3,color='g')
    ax.add_patch(circ)

for xx,yy in zip(new[:,0],new[:,1]):
    circ = Circle((xx,yy),10,color='greenyellow',fill=False)
    ax.add_patch(circ)

plt.imshow(img)
plt.savefig("4_2_1.jpg")
plt.close()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(cad_pts[:,0], cad_pts[:,1], cad_pts[:,2], triangles=cad_line, edgecolor=[[0,0,1]], linewidth=0.05, alpha=0, shade=False)
ax.view_init(elev=45., azim=220)
plt.savefig("4_2_2.jpg")
plt.close()

cad_R = np.matmul(R, cad_pts.T).T
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(cad_R[:,0], cad_R[:,1], cad_R[:,2], triangles=cad_line, edgecolor=[[0,0.5,0.5]], linewidth=0.05, alpha=0, shade=False)
ax.view_init(elev=1., azim=255)
plt.savefig("4_2_3.jpg")
plt.close()


cat_pts4 = np.concatenate([cad_pts, np.ones((cad_pts.shape[0], 1))], axis=1)
cat_pts4 = np.matmul(P, cat_pts4.T).T
cat_pts4 = cat_pts4/cat_pts4[:, -1:]
fig = plt.figure()
plt.imshow(img)
triang = mtri.Triangulation(cat_pts4[:,0], cat_pts4[:,1], cad_line)
z = np.zeros((cat_pts4.shape[0],))
plt.tricontourf(triang, z, colors="r", alpha=0.5)
plt.savefig("4_2_4.jpg")
plt.close()