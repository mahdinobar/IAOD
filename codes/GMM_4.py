"""
GMM_3

applying point cloud
"""

#hyperparameters
pca1=6
pca2=4
k1=5
cluster_span=1
#
# # import the necessary packages
import argparse
import cv2
import freenect
#
# # initialize the list of reference points and boolean indicating
# # whether cropping is being performed or not
# ref_point = []
# cropping = False
#
# def shape_selection(event, x, y, flags, param):
# 	# grab references to the global variables
# 	global ref_point, cropping
#
# 	# if the left mouse button was clicked, record the starting
# 	# (x, y) coordinates and indicate that cropping is being
# 	# performed
# 	if event == cv2.EVENT_LBUTTONDOWN:
# 		ref_point = [(x, y)]
# 		cropping = True
#
# 	# check to see if the left mouse button was released
# 	elif event == cv2.EVENT_LBUTTONUP:
# 		# record the ending (x, y) coordinates and indicate that
# 		# the cropping operation is finished
# 		ref_point.append((x, y))
# 		cropping = False
#
# 		# draw a rectangle around the region of interest
# 		cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
# 		cv2.imshow("image", image)
#
# def get_video():
#     array, _ = freenect.sync_get_video()
#     array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
#     return array
#
# # construct the argument parser and parse the arguments
# # ap = argparse.ArgumentParser()
# # ap.add_argument("-i", "--image", required=True, help="Path to the image")
# # args = vars(ap.parse_args())
#
# # load the image, clone it, and setup the mouse callback function
# #image = cv2.imread(args["image"])
# image=get_video()
#
# clone = image.copy()
# cv2.namedWindow("image")
# cv2.setMouseCallback("image", shape_selection)
#
# # keep looping until the 'q' key is pressed
# while True:
# 	# display the image and wait for a keypress
# 	cv2.imshow("image", image)
# 	key = cv2.waitKey(1) & 0xFF
#
# 	# if the 'r' key is pressed, reset the cropping region
# 	if key == ord("r"):
# 		image = clone.copy()
#
# 	# if the 'c' key is pressed, break from the loop
# 	elif key == ord("c"):
# 		break
#
# # if there are two reference points, then crop the region of interest
# # from teh image and display it
# if len(ref_point) == 2:
# 	crop_img = clone[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
# 	cv2.imshow("crop_img", crop_img)
# 	cv2.waitKey(0)
#
# # close all open windows
# cv2.destroyAllWindows()
# #Finish Cropping Part---------------------------------------------------------------------------------------------------
#
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
#
#
#
# def get_video():
#     array, _ = freenect.sync_get_video()
#     array=array[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
#     # array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
#     array = array.astype(np.uint8)
#     return array
#
#
# # function to get depth image from kinect
# def get_depth():
#     array, _ = freenect.sync_get_depth(format=freenect.DEPTH_REGISTERED)
#     array = array[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
#     # print (array.shape)
#     array = array.astype(np.uint8)
#     return array
#
#
# rgb = get_video()
# depth = get_depth()
#
#
# rgb = open3d.Image(rgb)
# depth = open3d.Image(depth)
#
# rgbd_image = open3d.create_rgbd_image_from_color_and_depth(rgb, depth,convert_rgb_to_intensity = False)
#
# fx = 594.21
# fy = 591.04
# a = -0.0030711
# b = 3.3309495
# cx = 339.5
# cy = 242.7
#
# intrinsic = open3d.PinholeCameraIntrinsic( 640, 480, fx,  fy,  cx,  cy)
# #
# # pcd = open3d.create_point_cloud_from_rgbd_image(rgbd_image, intrinsic)
#
# # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
# #
#
# # open3d.write_point_cloud('pcd_gmm_raw_data_more_space.ply', pcd)
pcd=open3d.read_point_cloud('pcd_gmm_raw_data_more_space.ply')
pcd_points = np.asarray(pcd.points)
pcd_colours = np.asarray(pcd.colors)
print(pcd_colours[1000,:])
open3d.draw_geometries([pcd])
X = np.hstack((pcd_points, pcd_colours))

np.save("raw_DATA_last_run_GMM_3.npy",X)
# X=np.load("DATA_11march.npy")
X_copy=np.copy(X)



# X=np.load("DATA_11march.npy")

#Start Standardize features==============================================================================
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X=scaler.transform(X)
# End  Standardize features==============================================================================

#Start PCA================================================================================================
from sklearn.decomposition import PCA
#X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# pca1=6
# pca = PCA(n_components=pca1)
# pca.fit(X)
# print(pca.explained_variance_ratio_)
# X = pca.transform(X)
# End PCA================================================================================================
# from tempfile import TemporaryFile
# X_11March = TemporaryFile()
# np.save("DATA_11march.npy",X)
# X=np.load("DATA_11march.npy")

# while 1:
#     cv2.imshow("imageX", image)
#     key = cv2.waitKey(1)
#     if key == 27:
#         break

#End (x,y) creation========================================================================================


print(__doc__)

# Number of samples per component


# Generate random sample, two components
# np.random.seed(0)
# C = np.array([[0., -0.1, 5], [1.7, .4, .4],[1.7, .4, .4]])
# X = np.r_[np.dot(np.random.randn(n_samples, 3), C),
#           .7 * np.random.randn(n_samples, 3) + np.array([-6, 3, 9])]
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
cv_types = ['spherical']
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
X = X[:,0:2]
#Start Standardize features==============================================================================
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X=scaler.transform(X)
# End  Standardize features==============================================================================

#Start PCA after background removal================================================================================================
from sklearn.decomposition import PCA
#X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# pca2=6
# pca = PCA(n_components=pca2)
# print("second PCA number of components= ",pca2)
# pca.fit(X)
# print("pca.explained_variance_ratio_ for PCA after background removal: ",pca.explained_variance_ratio_)
# X = pca.transform(X)
# End PCA after background removal================================================================================================


# while 1:
#     cv2.imshow("imageX", image)
#     key = cv2.waitKey(1)
#     if key == 27:
#         break

#End (x,y) creation========================================================================================


from sklearn import metrics
lowest_bic = np.infty
bic = []
lowest_SC = np.infty
lowest_CH = np.infty
lowest_DB = np.infty
SC=[]
CH=[]
# DB=[]
# k1=2
n_components_range = range(k1, k1+cluster_span)
# cv_types = ['spherical','diag','full']
cv_types = ['spherical']
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type=cv_type,init_params='random')
        gmm.fit(X)
        print("GMM number of gaussians(k)= ",n_components)
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
color_iter = itertools.cycle(['blue','yellow','navy', 'turquoise', 'cornflowerblue',
                              'darkorange','red','green','cyan','black'])
# clf = best_gmm
# clf_SC=best_gmm_SC
clf_CH=best_gmm_CH

#End background removal----------------------------------------------------------------------------------

# clf_DB=best_gmm_DB
bars = []



# Plot the BIC scores-----------------------------------------------------------------------------------
# plt.figure(figsize=(8, 6))
# spl = plt.subplot(2, 1, 1)
# for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
#     xpos = np.array(n_components_range) + .2 * (i - 2)
#     bars.append(plt.bar(xpos, bic[i * len(n_components_range):
#                                   (i + 1) * len(n_components_range)],
#                         width=.2, color=color))
# plt.xticks(n_components_range)
# plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
# plt.title('BIC score per model')
# xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
#     .2 * np.floor(bic.argmin() / len(n_components_range))
# plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
# spl.set_xlabel('Number of components')
# spl.legend([b[0] for b in bars], cv_types)



# Plot the BIC winner--------------------------------------------------------------------------------
# splot = plt.subplot(2, 1, 2)
# Y_ = clf.predict(X)
#
# K_counter=0
# # X=np.concatenate((c2c1,X),axis=1)
# for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_,
#                                            color_iter)):
#     v, w = linalg.eigh(cov)
#     K_counter=K_counter+1
#     # print(K_counter)
#     if not np.any(Y_ == i):
#         continue
#     plt.scatter(X_copy[Y_ == i, 0], -X_copy[Y_ == i, 1], .8, color=color)
#
#     # Plot an ellipse to show the Gaussian component
#     angle = np.arctan2(w[1][0], w[0][0])
#     angle = 180. * angle / np.pi  # convert to degrees
#     v = 2. * np.sqrt(2.) * np.sqrt(v)
#     ell = mpl.patches.Ellipse(mean, v[0], v[1],  angle, color=color)
#     ell.set_clip_box(splot.bbox)
#     ell.set_alpha(.5)
#     splot.add_artist(ell)

# plt.xticks(())
# plt.yticks(())
# plt.title('Selected GMM: * model, * components')
# plt.subplots_adjust(hspace=.35, bottom=.02)
# plt.show()


# bars_SC = []
# # Plot the SC scores-------------------------------------------------------------------------
# # plt.figure(figsize=(8, 6))
# spl = plt.subplot(2, 1, 1)
# for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
#     xpos_SC = np.array(n_components_range) + .2 * (i - 2)
#     bars_SC.append(plt.bar(xpos_SC, SC[i * len(n_components_range):
#                                   (i + 1) * len(n_components_range)],
#                         width=.2, color=color))
# plt.xticks(n_components_range)
# plt.ylim([SC.min() * 1.01 - .01 * SC.max(), SC.max()])
# plt.title('SC score per model')
# xpos_SC = np.mod(SC.argmin(), len(n_components_range)) + .65 +\
#     .2 * np.floor(SC.argmin() / len(n_components_range))
# plt.text(xpos_SC, SC.min() * 0.97 + .03 * SC.max(), '*', fontsize=14)
# spl.set_xlabel('Number of components')
# spl.legend([b[0] for b in bars_SC], cv_types)
#
# #End plot SC scores--------------------------------------------------------------------------
#
# #Start Plot the SC winner-------------------------------------------------------------------------
# splot2 = plt.subplot(2, 1, 2)
# Y_ = clf_SC.predict(X)
# # X_copy=c2c1
# K_counter=0
# # X=np.concatenate((c2c1,X),axis=1)
# for i, (mean, cov, color) in enumerate(zip(clf_SC.means_, clf_SC.covariances_,
#                                            color_iter)):
#     v, w = linalg.eigh(cov)
#     K_counter=K_counter+1
#     print(K_counter)
#     if not np.any(Y_ == i):
#         continue
#     plt.scatter(X_copy[Y_ == i, 0], -X_copy[Y_ == i, 1], .8, color=color)
#
#     # Plot an ellipse to show the Gaussian component
#     angle = np.arctan2(w[1][0], w[0][0])
#     angle = 180. * angle / np.pi  # convert to degrees
#     v = 2. * np.sqrt(2.) * np.sqrt(v)
#     ell = mpl.patches.Ellipse(mean, v[0], v[1],  angle, color=color)
#     ell.set_clip_box(splot2.bbox)
#     ell.set_alpha(.5)
#     splot2.add_artist(ell)
#
# plt.xticks(())
# plt.yticks(())
# plt.title('Selected GMM (SC winner): * model, * components')
# plt.subplots_adjust(hspace=.35, bottom=.02)
# plt.show()
# print('END GMM_1')
# #End Plot the SC winner-------------------------------------------------------------------------

bars_CH = []
# Plot the CH scores-------------------------------------------------------------------------
# plt.figure(figsize=(8, 6))
spl = plt.subplot(2, 1, 1)
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos_CH = np.array(n_components_range) + .2 * (i - 2)
    bars_CH.append(plt.bar(xpos_CH, CH[i * len(n_components_range):
                                  (i + 1) * len(n_components_range)],
                        width=.2, color=color))
plt.xticks(n_components_range)
plt.ylim([CH.min() * 1.01 - .01 * CH.max(), CH.max()])
plt.title('CH score per model')
# xpos_CH = np.mod(CH.argmin(), len(n_components_range)) + .65 +\
#     .2 * np.floor(CH.argmin() / len(n_components_range))
# plt.text(xpos_CH, CH.min() * 0.97 + .03 * CH.max(), '*', fontsize=14)
spl.set_xlabel('Number of components')
spl.legend([b[0] for b in bars_CH], cv_types)

#End plot CH scores--------------------------------------------------------------------------

#Start Plot the CH winner-------------------------------------------------------------------------
splot2 = plt.subplot(2, 1, 2)
Y_ = clf_CH.predict(X)
# X_copy=c2c1
K_counter=0
# X=np.concatenate((c2c1,X),axis=1)
# for i, (mean, cov, color) in enumerate(zip(clf_CH.means_, clf_CH.covariances_,
#                                            color_iter)):
#     v, w = linalg.eigh(cov)
#     K_counter=K_counter+1
#
#     if not np.any(Y_ == i):
#         continue
#     plt.scatter(X_copy[Y_ == i, 0], -X_copy[Y_ == i, 1], .8, color=color)
#
#     # Plot an ellipse to show the Gaussian component
#     angle = np.arctan2(w[1][0], w[0][0])
#     angle = 180. * angle / np.pi  # convert to degrees
#     v = 2. * np.sqrt(2.) * np.sqrt(v)
#     ell = mpl.patches.Ellipse(mean, v[0], v[1],  angle, color=color)
#     ell.set_clip_box(splot2.bbox)
#     ell.set_alpha(.5)
#     splot2.add_artist(ell)
print("The optimal number of clusters= ", K_counter)
plt.xticks(())
plt.yticks(())
plt.title('Selected GMM (CH winner): * model, * components')
plt.subplots_adjust(hspace=.35, bottom=.02)
plt.show()
print('END GMM_1')
#End Plot the CH winner-------------------------------------------------------------------------

# bars_DB = []
# # Plot the DB scores-------------------------------------------------------------------------
# # plt.figure(figsize=(8, 6))
# spl = plt.subplot(2, 1, 1)
# for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
#     xpos_DB = np.array(n_components_range) + .2 * (i - 2)
#     bars_DB.append(plt.bar(xpos_DB, DB[i * len(n_components_range):
#                                   (i + 1) * len(n_components_range)],
#                         width=.2, color=color))
# plt.xticks(n_components_range)
# plt.ylim([DB.min() * 1.01 - .01 * DB.max(), DB.max()])
# plt.title('DB score per model')
# xpos_DB = np.mod(DB.argmin(), len(n_components_range)) + .65 +\
#     .2 * np.floor(DB.argmin() / len(n_components_range))
# plt.text(xpos_DB, DB.min() * 0.97 + .03 * DB.max(), '*', fontsize=14)
# spl.set_xlabel('Number of components')
# spl.legend([b[0] for b in bars_DB], cv_types)
#
# #End plot DB scores--------------------------------------------------------------------------
#
# #Start Plot the DB winner-------------------------------------------------------------------------
# splot2 = plt.subplot(2, 1, 2)
# Y_ = clf_DB.predict(X)
# # X_copy=c2c1
# K_counter=0
# # X=np.concatenate((c2c1,X),axis=1)
# for i, (mean, cov, color) in enumerate(zip(clf_DB.means_, clf_DB.covariances_,
#                                            color_iter)):
#     v, w = linalg.eigh(cov)
#     K_counter=K_counter+1
#     print(K_counter)
#     if not np.any(Y_ == i):
#         continue
#     plt.scatter(X_copy[Y_ == i, 0], -X_copy[Y_ == i, 1], .8, color=color)
#
#     # Plot an ellipse to show the Gaussian component
#     angle = np.arctan2(w[1][0], w[0][0])
#     angle = 180. * angle / np.pi  # convert to degrees
#     v = 2. * np.sqrt(2.) * np.sqrt(v)
#     ell = mpl.patches.Ellipse(mean, v[0], v[1],  angle, color=color)
#     ell.set_clip_box(splot2.bbox)
#     ell.set_alpha(.5)
#     splot2.add_artist(ell)
#
# plt.xticks(())
# plt.yticks(())
# plt.title('Selected GMM (DB winner): * model, * components')
# plt.subplots_adjust(hspace=.35, bottom=.02)
# plt.show()
# print('END GMM_1')
# #End Plot the DB winner-------------------------------------------------------------------------

from open3d import *
import open3d
color=np.array([[204,0,0],[0,204,0],[0,0,204],[255,0,127],[255,255,0],[127,0,255],[255,128,0],[102,51,0],[255,153,153],[153,255,255],[0,102,102]])/255
for i in range (np.unique(Y_).min(),np.unique(Y_).max()+1):
    # plt.scatter(X_copy[Y_ == i, 0], X_copy[Y_ == i, 1], .8, color=color)
    X_copy[Y_ == i, 3:6]=color[i]

pcd.colors = Vector3dVector(X_copy[:, 3:6])
pcd.points = Vector3dVector(X_copy[:, 0:3])
open3d.draw_geometries([pcd])


