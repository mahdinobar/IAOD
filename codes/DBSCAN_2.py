"""
DBSCAN_2

apply point cloud
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

intrinsic = open3d.PinholeCameraIntrinsic( 640, 480, fx,  fy,  cx,  cy)

# pcd = open3d.create_point_cloud_from_rgbd_image(rgbd_image, intrinsic)

# pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
pcd=open3d.read_point_cloud('pcd_gmm_raw_data.ply')
pcd_points = np.asarray(pcd.points)
pcd_colours = np.asarray(pcd.colors)

print(pcd_colours[1000,:])
open3d.draw_geometries([pcd])
X = np.hstack((pcd_points, pcd_colours))


np.save("raw_DATA_last_run_DBSCAN_3.npy",X)
# X=np.load("DATA_11march.npy")
X_copy=np.copy(X)

#Start Standardize features==============================================================================
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X=scaler.transform(X)
# End  Standardize features==============================================================================

#Start background detection with GMM--------------------------------------------------------------------------------
from sklearn import metrics
import numpy as np
import itertools
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import mixture
import freenect
import cv2

lowest_bic = np.infty
bic = []
lowest_SC = np.infty
lowest_CH = np.infty
lowest_DB = np.infty
SC=[]
CH=[]
# DB=[]
# k1=2
n_components_range = range(2, 3)
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
# if X[Y_==0,2].mean()>X[Y_==1,2].mean():
#     X[Y_ == 1, 2] = X[Y_ == 1, 2].min()
# else:
#     X[Y_ == 0, 2] = X[Y_ == 0, 2].min()
if X[Y_==0,2].mean()>X[Y_==1,2].mean():
    X = X[Y_ == 0, :]
    X_copy=X_copy[Y_ == 0, :]
else:
    X = X[Y_ == 1, :]
    X_copy = X_copy[Y_ == 1, :]
# X[Y_==0,:]=X[Y_==1,:].min()
#End background removal--------------------------------------------------------------------------------

# X = np.hstack((X[:,0:2], X[:,5:4]))

#Start Standardize features==============================================================================
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X=scaler.transform(X)
# End  Standardize features==============================================================================

#Start PCA================================================================================================
from sklearn.decomposition import PCA
#X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca = PCA(n_components=6)
pca.fit(X)
print(pca.explained_variance_ratio_)
X = pca.transform(X)
# End PCA================================================================================================






print(__doc__)

import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler


# #############################################################################
# Generate sample data
# centers = [[1, 1], [-1, -1], [1, -1]]
# X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
#                             random_state=0)

# X=np.load("raw_DATA_last_run_GMM_2.npy")

# X = StandardScaler().fit_transform(X)

# #############################################################################
# Compute DBSCAN
CH=[]
lowest_CH = np.infty
silhouette=[]
lowest_silhouette = np.infty
for epsilon in np.arange(0.2,0.21,0.01):
    db = DBSCAN(eps=epsilon, min_samples=150).fit(X)
    # core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    # core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    # CH.append(metrics.calinski_harabaz_score(X, labels))
    # if CH[-1] < lowest_CH:
    #     lowest_CH = CH[-1]
    #     best_dbscan_CH = db
    #     best_epsilon=epsilon
    silhouette.append(metrics.silhouette_score(X, labels, metric='euclidean'))
    if silhouette[-1] < lowest_silhouette:
        lowest_silhouette = silhouette[-1]
        best_dbscan_silhouette = db
        best_epsilon_silhouette = epsilon

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# print("DBSCAN best epsilon based on CH= ",best_epsilon)
print("DBSCAN best epsilon based on bic= ",best_epsilon_silhouette)
print("DBSCAN estimated number of clusters= ",n_clusters_)
# n_noise_ = list(labels).count(-1)

# print('Estimated number of clusters: %d' % n_clusters_)
# print('Estimated number of noise points: %d' % n_noise_)
# print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
# print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
# print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
# print("Adjusted Rand Index: %0.3f"
#       % metrics.adjusted_rand_score(labels_true, labels))
# print("Adjusted Mutual Information: %0.3f"
#       % metrics.adjusted_mutual_info_score(labels_true, labels))
# print("Silhouette Coefficient: %0.3f"
#       % metrics.silhouette_score(X, labels))

# #############################################################################
# Plot result
import matplotlib.pyplot as plt
#
# # Black removed and is used for noise instead.
# unique_labels = set(labels)
# from sklearn import metrics
# import itertools
# color_iter=itertools.cycle(['blue','yellow','navy', 'turquoise', 'cornflowerblue',
#                               'darkorange','red','green','cyan'])
# for k, color in zip(unique_labels, color_iter):
#     if k == -1:
#         # Black used for noise.
#         color = ('black')
#
#     print("unique_labels= ",k,";   color= ",color)
#
#     class_member_mask = (labels == k)
#
#     xy = X_copy[class_member_mask & core_samples_mask]
#     plt.plot(xy[:, 0], xy[:, 1], 'o', color=color,
#              markersize=5)
#
#     xy = X_copy[class_member_mask & ~core_samples_mask]
#     plt.plot(xy[:, 0], xy[:, 1], 'o', color=color,
#              markersize=2)
#
# plt.title('Estimated number of clusters: %d' % n_clusters_)
# plt.show()



from open3d import *
import open3d
color=np.array([[0,0,0],[204,0,0],[0,204,0],[0,0,204],[255,0,127],[255,255,0],[127,0,255],[255,128,0],[102,51,0],[255,153,153],[153,255,255],[0,102,102]])/255
for i in range (np.unique(labels).min(),np.unique(labels).max()+1):
    # plt.scatter(X_copy[Y_ == i, 0], X_copy[Y_ == i, 1], .8, color=color)

    X_copy[labels == i, 3:6]=color[i+1]

pcd.colors = Vector3dVector(X_copy[:, 3:6])
pcd.points = Vector3dVector(X_copy[:, 0:3])
open3d.draw_geometries([pcd])