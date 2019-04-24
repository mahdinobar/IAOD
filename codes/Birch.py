"""
Birch clustering
"""
# import the necessary packages
import argparse
import cv2
import freenect

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
ref_point = []
cropping = False

def shape_selection(event, x, y, flags, param):
	# grab references to the global variables
	global ref_point, cropping

	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		ref_point = [(x, y)]
		cropping = True

	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		ref_point.append((x, y))
		cropping = False

		# draw a rectangle around the region of interest
		cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
		cv2.imshow("image", image)

def get_video():
    array, _ = freenect.sync_get_video()
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    return array

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="Path to the image")
# args = vars(ap.parse_args())

# load the image, clone it, and setup the mouse callback function
#image = cv2.imread(args["image"])
image=get_video()

clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", shape_selection)

# keep looping until the 'q' key is pressed
while True:
	# display the image and wait for a keypress
	cv2.imshow("image", image)
	key = cv2.waitKey(1) & 0xFF

	# if the 'r' key is pressed, reset the cropping region
	if key == ord("r"):
		image = clone.copy()

	# if the 'c' key is pressed, break from the loop
	elif key == ord("c"):
		break

# if there are two reference points, then crop the region of interest
# from teh image and display it
if len(ref_point) == 2:
	crop_img = clone[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
	cv2.imshow("crop_img", crop_img)
	cv2.waitKey(0)

# close all open windows
cv2.destroyAllWindows()
#Finish Cropping Part---------------------------------------------------------------------------------------------------

#https://www.learnopencv.com/hand-keypoint-detection-using-deep-learning-and-opencv/
import time
import freenect
import cv2
import numpy as np

#modified version to 3D data
import numpy as np
import itertools
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import mixture
import freenect
import cv2
import freenect
import cv2
import open3d
import numpy as np
import matplotlib.pyplot as plt



def get_video():
    array, _ = freenect.sync_get_video()
    array=array[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
    # array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    array = array.astype(np.uint8)
    return array


# function to get depth image from kinect
def get_depth():
    array, _ = freenect.sync_get_depth(format=freenect.DEPTH_REGISTERED)
    array = array[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
    # print (array.shape)
    array = array.astype(np.uint8)
    return array


rgb = get_video()
depth = get_depth()


rgb = open3d.Image(rgb)
depth = open3d.Image(depth)

rgbd_image = open3d.create_rgbd_image_from_color_and_depth(rgb, depth,convert_rgb_to_intensity = False)

fx = 594.21
fy = 591.04
a = -0.0030711
b = 3.3309495
cx = 339.5
cy = 242.7

intrinsic = open3d.PinholeCameraIntrinsic(640, 480, fx,  fy,  cx,  cy)

# pcd = open3d.create_point_cloud_from_rgbd_image(rgbd_image, intrinsic)
#
# pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])





# open3d.write_point_cloud('pcd_birch_raw_data.ply', pcd)
pcd=open3d.read_point_cloud('pcd_gmm_raw_data.ply')
pcd_points = np.asarray(pcd.points)
pcd_colours = np.asarray(pcd.colors)
open3d.draw_geometries([pcd])
X = np.hstack((pcd_points, pcd_colours))


np.save("raw_DATA_last_run_DBSCAN_3.npy",X)
# X=np.load("DATA_11march.npy")
X_copy=np.copy(X)


print(__doc__)

import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs

#Start Standardize features==============================================================================
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X=scaler.transform(X)
# End  Standardize features==============================================================================
#Start PCA================================================================================================
from sklearn.decomposition import PCA
#X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# pca1=4
# pca = PCA(n_components=pca1)
# pca.fit(X)
# print("first PCA number of components= ",pca1)
# print(pca.explained_variance_ratio_)
# X = pca.transform(X)
# End PCA================================================================================================

#Start background detection with GMM--------------------------------------------------------------------------------
from sklearn import metrics
from sklearn import mixture
import itertools
lowest_bic = np.infty
bic = []
lowest_SC = np.infty
lowest_CH = np.infty
lowest_DB = np.infty
SC=[]
CH=[]
# DB=[]
k1=2
n_components_range = range(k1, k1+1)
# cv_types = ['spherical','diag','full']
cv_types = ['full']
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type=cv_type)
        gmm.fit(X)
        print("backgroundremoval GMM number of gaussians(k)= ", n_components)
        #Start Silhouette Coefficient metric-------------------------------------
        # labels=gmm.predict(X)
        # from sklearn import metrics
        # SC.append(metrics.silhouette_score(X, gmm.predict(X), metric='euclidean'))
        # if SC[-1] < lowest_SC:
        #     lowest_SC = SC[-1]
        #     best_gmm_SC = gmm
        #End Silhouette Coefficient metric test-------------------------------------
        #Start Davies-Bouldin Index-------------------------------------
        # labels=gmm.predict(X)
        # from sklearn import metrics
        CH.append(metrics.calinski_harabaz_score(X, gmm.predict(X)))
        if CH[-1] < lowest_CH:
            lowest_CH = CH[-1]
            best_gmm_CH = gmm
        # DB.append(metrics.davies_bouldin_score(X, gmm.predict(X)))
        # if DB[-1] < lowest_DB:
        #     lowest_DB = DB[-1]
        #     best_gmm_DB = gmm
        #End Davies-Bouldin Index-------------------------------------
        # bic.append(gmm.bic(X))
        # if bic[-1] < lowest_bic:
        #     lowest_bic = bic[-1]
        #     best_gmm = gmm

# bic = np.array(bic)
# SC=np.array(SC)
CH=np.array(CH)
# DB=np.array(DB)
color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                              'darkorange','red','green','blue','yellow','cyan','black'])
# clf = best_gmm
# clf_SC=best_gmm_SC
clf_CH=best_gmm_CH
#End background detection with GMM--------------------------------------------------------------------------------

#Start background removal--------------------------------------------------------------------------------
Y_ = clf_CH.predict(X)
if X[Y_==0,2].mean()>X[Y_==1,2].mean():
    X = X[Y_ == 0, :]
    X_copy=X_copy[Y_ == 0, :]
else:
    X = X[Y_ == 1, :]
    X_copy = X_copy[Y_ == 1, :]
#End background removal--------------------------------------------------------------------------------
X = np.hstack((X[:,0:2], X[:,3:6]))

#Start Standardize features==============================================================================
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X=scaler.transform(X)
# End  Standardize features==============================================================================

#Start PCA after background removal================================================================================================
from sklearn.decomposition import PCA
#X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca2=4
pca = PCA(n_components=pca2)
print("second PCA number of components= ",pca2)
pca.fit(X)
print("pca.explained_variance_ratio_ for PCA after background removal: ",pca.explained_variance_ratio_)
X = pca.transform(X)
# End PCA after background removal================================================================================================




# The following bandwidth can be automatically detected using

from sklearn.cluster import Birch
brc = Birch(branching_factor=4, n_clusters=8, threshold=0.5,
            compute_labels=True)
brc.fit(X)
labelsBCHH=brc.predict(X)




# labels_unique = np.unique(labels)
labels_unique = np.unique(labelsBCHH)
n_clusters_ = len(labels_unique)

print("Birch number of estimated clusters : %d" % n_clusters_)

# #############################################################################
# Plot result
# import matplotlib.pyplot as plt
# from itertools import cycle
#
# plt.figure(1)
# plt.clf()
#
# colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
# for k, col in zip(range(n_clusters_), colors):
#     my_members = labels == k
#     # cluster_center = cluster_centers[k]
#     plt.plot(X_copy[my_members, 0], X_copy[my_members, 1], col + '.')
#     # plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
#     #          markeredgecolor='k', markersize=14)
# plt.title('Estimated number of clusters: %d' % n_clusters_)
# plt.show()

from open3d import *
import open3d
color=np.array([[0,0,0],[204,0,0],[0,204,0],[0,0,204],[255,0,127],[255,255,0],[127,0,255],[255,128,0],[102,51,0],[255,153,153],[153,255,255],[0,102,102]])/255
for i in range (np.unique(labelsBCHH).min(),np.unique(labelsBCHH).max()+1):
    # plt.scatter(X_copy[Y_ == i, 0], X_copy[Y_ == i, 1], .8, color=color)

    X_copy[labelsBCHH == i, 3:6]=color[i+1]

pcd.colors = Vector3dVector(X_copy[:, 3:6])
pcd.points = Vector3dVector(X_copy[:, 0:3])
open3d.draw_geometries([pcd])