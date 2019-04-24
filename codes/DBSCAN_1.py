"""
DBSCAN_1

it needs modification
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



def get_video():
    array, _ = freenect.sync_get_video()
    array=array[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    return array


# function to get depth image from kinect
def get_depth():
    array, _ = freenect.sync_get_depth(format=freenect.DEPTH_REGISTERED)
    array = array[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
    # print (array.shape)
    array = array.astype(np.uint8)
    return array

# def get_video():
#     array, _ = freenect.sync_get_video()
#     array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
#     return array
#
# def get_depth():
#     array, _ = freenect.sync_get_depth()
#     # array = array[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
#     # print (array.shape)
#     array = array.astype(np.uint8)
#     return array

image=get_video()
depth=get_depth()

#Start missing depth data replacement with mean of available data------------------------------------------------
depth_copy=np.copy(depth)
nul_pixel_counter=0
for r in range (0,depth.shape[0]):
    for c in range (0,depth.shape[1]):
        if depth[r][c]>220:
            depth_copy[r][c]=0
            nul_pixel_counter=nul_pixel_counter+1

for r in range (0,depth.shape[0]):
    for c in range (0,depth.shape[1]):
        if depth[r][c]>220:
            depth[r][c]=depth_copy.mean()*depth_copy.size/(depth_copy.size-nul_pixel_counter)
#End missing depth data replacement with mean of available data------------------------------------------------



n_samples = image.shape[0]*image.shape[1]
X1=image.reshape((n_samples,3))
X2=depth.reshape((n_samples,1))
X3=np.concatenate((X2,X1),axis=1)
# X3=X2
# X3=X1
#Start (x,y) creation======================================================================================
# a=np.empty([image.shape[1],1])
# h=np.empty([image.shape[1],1])
# h.fill(0)
# c1=h
# # c2=np.empty([640,1])
# b=np.arange(image.shape[1])
# c2=b
# for k in range(1,image.shape[0]):
#     a.fill(k)
#     c1=np.concatenate((c1,a),axis=0)
#     b=np.arange(image.shape[1])
#     c2=np.concatenate((c2,b),axis=0)
# # c2=np.transpose(c2)
# c2=c2.reshape((image.shape[0]*image.shape[1],1))
# c2c1=np.concatenate((c2,c1),axis=1)
# X=np.concatenate((c2c1,X3),axis=1)

a=np.empty([image.shape[1],1])
h=np.empty([image.shape[1],1])
h.fill(ref_point[0][1])
c1=h #c1 is for y
# c2=np.empty([640,1])
b=np.arange(image.shape[1])+ref_point[0][0]
c2=b #c2 is for x
for k in range(1+ref_point[0][1],ref_point[1][1]):
    a.fill(k)
    c1=np.concatenate((c1,a),axis=0)
    # b=np.arange(image.shape[1])+ref_point[0][0]
    c2=np.concatenate((c2,b),axis=0)
# c2=np.transpose(c2)
c2=c2.reshape((n_samples,1))
c2c1=np.concatenate((c2,c1),axis=1)#c2c1 contaimns corresponding (x,y) coordinates of each sample on frame
X=np.concatenate((c2c1,X3),axis=1)
# from tempfile import TemporaryFile
# X_11March = TemporaryFile()
np.save("raw_DATA_last_run_DBSCAN_1.npy",X)
# X=np.load("DATA_11march.npy")
X_copy=np.copy(X)
#Start Standardize features==============================================================================
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X=scaler.transform(X)
# End  Standardize features==============================================================================

#Start PCA================================================================================================
from sklearn.decomposition import PCA
#X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca = PCA(n_components=4)
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

db = DBSCAN(eps=0.8, min_samples=100).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

# print('Estimated number of clusters: %d' % n_clusters_)
# print('Estimated number of noise points: %d' % n_noise_)
# print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
# print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
# print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
# print("Adjusted Rand Index: %0.3f"
#       % metrics.adjusted_rand_score(labels_true, labels))
# print("Adjusted Mutual Information: %0.3f"
#       % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels))

# #############################################################################
# Plot result
import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
unique_labels = set(labels)
from sklearn import metrics
import itertools
color_iter=itertools.cycle(['blue','yellow','navy', 'turquoise', 'cornflowerblue',
                              'darkorange','red','green','cyan'])
for k, color in zip(unique_labels, color_iter):
    if k == -1:
        # Black used for noise.
        color = ('black')

    print("unique_labels= ",k,";   color= ",color)

    class_member_mask = (labels == k)

    xy = X_copy[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', color=color,
             markersize=5)

    xy = X_copy[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', color=color,
             markersize=2)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
