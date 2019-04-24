"""
GMM , K=, with initial condition without background,3 features, remove background, no PCA:

this scripts works for one objects("GMM_6_image_1_objects.npy" and "GMM_6_depth_1_objects.npy") matching with 7 object ground truth we have from GMM_6
it works for all 7 objects except object 7
"""

from open3d import *
import cv2 as cv
import open3d
import numpy as np
import itertools
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import mixture
import freenect
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import math


print(__doc__)

image =np.load("GMM_6_image_1_objects.npy")
depth=np.load("GMM_6_depth_1_objects.npy")
imgray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
kernel = np.ones((14,14),np.uint8)
imgray = cv.erode(imgray,kernel,iterations = 1)
ret, thresh = cv.threshold(imgray, 95, 255, 0)
while 1:
    cv.imshow('thresh', thresh)
    key = cv.waitKey(1)
    if key==27:
        break
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE) #contours contains here the pixel index of the corresponding points
image_contours=np.copy(image)
cv.drawContours(image_contours, contours[1:], -1, (0,255,0), 3)
#find the mean value for the detected contours
contours_mean=np.empty([len(contours),2])
for n_contour in range(0,len(contours)):
    contours_mean[n_contour][:]=(contours[n_contour].mean(axis=0))
while 1:
    cv.imshow('Contours', image_contours)
    key = cv.waitKey(1)
    if key==27:
        break
# means_init=np.array()

contours_mean=np.around(contours_mean,decimals=0).astype(int)
image_contour_center=np.copy(image)
image_contour_center.fill(255.)
for c in range (0,contours_mean.shape[0]):
	image_contour_center[contours_mean[c][1]][contours_mean[c][0]]=image[contours_mean[c][1]][contours_mean[c][0]]



image_contour_center_3d = open3d.Image(image_contour_center)
depth_3d = open3d.Image(depth)
rgbd_image = open3d.create_rgbd_image_from_color_and_depth(image_contour_center_3d, depth_3d,convert_rgb_to_intensity = False)
fx = 594.21
fy = 591.04
a = -0.0030711
b = 3.3309495
cx = 339.5
cy = 242.7
intrinsic = open3d.PinholeCameraIntrinsic(depth.shape[1], depth.shape[0], fx,  fy,  cx,  cy)
pcd_contour_mean = open3d.create_point_cloud_from_rgbd_image(rgbd_image, intrinsic)
pcd_contour_mean.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
# np.asarray(pcd_contour_mean.points)[np.array([[1], [5], [0]]).transpose()]
# contours_mean_pcd_xyz=np.asarray(pcd_contour_mean.points)[np.asarray(pcd_contour_mean.colors).argsort(axis=0)[0:contours_mean.shape[0],0]]
# contours_mean_pcd_rgb=np.asarray(pcd_contour_mean.colors)[np.asarray(pcd_contour_mean.colors).argsort(axis=0)[0:contours_mean.shape[0],0]]

open3d.draw_geometries([pcd_contour_mean])










image_3d = open3d.Image(image)

rgbd_image = open3d.create_rgbd_image_from_color_and_depth(image_3d, depth_3d,convert_rgb_to_intensity = False)

fx = 594.21
fy = 591.04
a = -0.0030711
b = 3.3309495
cx = 339.5
cy = 242.7

intrinsic = open3d.PinholeCameraIntrinsic(depth.shape[1], depth.shape[0], fx,  fy,  cx,  cy)
#
pcd = open3d.create_point_cloud_from_rgbd_image(rgbd_image, intrinsic)

pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
#

# open3d.write_point_cloud('pcd_gmm_raw_data_more_space.ply', pcd)
# pcd=open3d.read_point_cloud('pcd_gmm_raw_data_more_space.ply')
pcd_points =np.asarray(pcd.points)
pcd_colours = np.asarray(pcd.colors)
print(pcd_colours[1000,:])
open3d.draw_geometries([pcd])
X = np.hstack((pcd_points, pcd_colours))



#Start Standardize features==============================================================================
scaler = StandardScaler()
scaler.fit(X)
X=scaler.transform(X)
X_init=X[np.asarray(pcd_contour_mean.colors).argsort(axis=0)[0:contours_mean.shape[0],0]]
# End  Standardize features==============================================================================

lowest_bic = np.infty
bic = []
lowest_SC = np.infty
lowest_CH = np.infty
lowest_DB = np.infty
SC=[]
CH=[]
k1=3
cluster_span=1
n_components_range = range(k1, k1+cluster_span)
cv_types = ['full']
X_copy=np.copy(X)

X=X[:,0:3]
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type=cv_type,
                                      means_init=X_init[:,0:3])
                                      # weights_init=np.ones(X_init.shape[0])/X_init.shape[0])
        gmm.fit(X)
        print("GMM number of gaussians(k)= ",n_components)


clf_CH=gmm


Y_ = clf_CH.predict(X)


color=np.array([[204,0,0],[0,204,0],[0,0,204],[255,0,127],[255,255,0],[127,0,255],[255,128,0],[102,51,0],[255,153,153],[153,255,255],[0,102,102]])/255
for i in range (np.unique(Y_).min(),np.unique(Y_).max()+1):
    # plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)
    X_copy[Y_ == i, 3:6]=color[i]

pcd.colors = Vector3dVector(X_copy[:, 3:6])
pcd.points = Vector3dVector(X_copy[:, 0:3])


part_covariance=np.load("part_covariance_GMM_6.npy")
part_mean=np.load("part_mean_GMM_6.npy")
covariance_distance=np.empty([part_covariance.shape[0],clf_CH.covariances_.shape[0]-1])
mean_distance=np.empty_like(covariance_distance)
kl_div=np.empty_like(covariance_distance)
Hellinger=np.empty_like(covariance_distance)

# clf_CH.covariances_[2]=c2=np.identity(3)
# part_covariance[2]=np.identity(3)
# clf_CH.means_[2]=np.zeros([3,])
# part_mean[2]=np.zeros([3,])

for c1 in range (0,part_covariance.shape[0]):
    for c2 in range (1,clf_CH.covariances_.shape[0]):
        covariance_distance[c1,c2-1]=(np.linalg.norm(clf_CH.covariances_[c2]-part_covariance[c1]))
        mean_distance[c1, c2-1] = (np.linalg.norm(clf_CH.means_[c2] - part_mean[c1]))
        kl_div[c1,c2-1]=0.5*(np.log(np.linalg.det(part_covariance[c1])/np.linalg.det(clf_CH.covariances_[c2]))
                           -part_mean.shape[1]
                           +np.trace(np.matmul(np.linalg.inv(part_covariance[c1]),clf_CH.covariances_[c2]))
                           +np.matmul(np.matmul((part_mean[c1]-clf_CH.means_[c2]),np.linalg.inv(part_covariance[c1]))
                                      ,(part_mean[c1]-clf_CH.means_[c2])))
        Hellinger[c1,c2-1]=math.sqrt(1-(np.linalg.det(clf_CH.covariances_[c2])*np.linalg.det(part_covariance[c1]))**(1/4)
                                   /np.linalg.det(clf_CH.covariances_[c2]/2+part_covariance[c1]/2)**(1/2)
                                   *math.exp(-np.matmul(np.matmul((part_mean[c1]-clf_CH.means_[c2]),
                                                                  np.linalg.inv(clf_CH.covariances_[c2]/2+part_covariance[c1]/2))
                                                        ,(part_mean[c1]-clf_CH.means_[c2]))/8))
kl_argsort=np.argsort(kl_div,axis=0)#to see if the result is correct

open3d.draw_geometries([pcd])