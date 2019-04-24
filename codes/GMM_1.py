import warnings
warnings.simplefilter("default", UserWarning)

# import the_module_that_warns
#
# #Start Cropping--------------------------------------------------------------------------------------------------------
#reference: https://www.codementor.io/innat_2k14/extract-a-particular-object-from-images-using-opencv-in-python-jfogyig5u
# Capture the mouse click events in Python and OpenCV
'''
-> draw shape on any image
-> reset shape on selection
-> crop the selection

run the code : python capture_events.py --image image_example.jpg

'''


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
    array, _ = freenect.sync_get_depth()
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
np.save("raw_DATA_last_run_GMM_1.npy",X)
# X=np.load("DATA_11march.npy")

#Start Standardize features==========================================================================
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X=scaler.transform(X)
# End  Standardize features==========================================================================
#Start PCA================================================================================================
from sklearn.decomposition import PCA
#X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca = PCA(n_components=6)
pca.fit(X)
print(pca.explained_variance_ratio_)
X = pca.transform(X)
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
np.random.seed(0)
# C = np.array([[0., -0.1, 5], [1.7, .4, .4],[1.7, .4, .4]])
# X = np.r_[np.dot(np.random.randn(n_samples, 3), C),
#           .7 * np.random.randn(n_samples, 3) + np.array([-6, 3, 9])]

from sklearn import metrics
lowest_bic = np.infty
bic = []
lowest_SC = np.infty
lowest_CH = np.infty
lowest_DB = np.infty
SC=[]
CH=[]
# DB=[]
k1=5
n_components_range = range(k1, k1+5)
# cv_types = ['spherical','diag','full']
cv_types = ['full']
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type=cv_type)
        gmm.fit(X)
        print(n_components)
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
        bic.append(gmm.bic(X))
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
# clf_DB=best_gmm_DB
bars = []
X_xy=c2c1


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
#     plt.scatter(X_xy[Y_ == i, 0], -X_xy[Y_ == i, 1], .8, color=color)
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
# # X_xy=c2c1
# K_counter=0
# # X=np.concatenate((c2c1,X),axis=1)
# for i, (mean, cov, color) in enumerate(zip(clf_SC.means_, clf_SC.covariances_,
#                                            color_iter)):
#     v, w = linalg.eigh(cov)
#     K_counter=K_counter+1
#     print(K_counter)
#     if not np.any(Y_ == i):
#         continue
#     plt.scatter(X_xy[Y_ == i, 0], -X_xy[Y_ == i, 1], .8, color=color)
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
xpos_CH = np.mod(CH.argmin(), len(n_components_range)) + .65 +\
    .2 * np.floor(CH.argmin() / len(n_components_range))
plt.text(xpos_CH, CH.min() * 0.97 + .03 * CH.max(), '*', fontsize=14)
spl.set_xlabel('Number of components')
spl.legend([b[0] for b in bars_CH], cv_types)

#End plot CH scores--------------------------------------------------------------------------

#Start Plot the CH winner-------------------------------------------------------------------------
splot2 = plt.subplot(2, 1, 2)
Y_ = clf_CH.predict(X)
# X_xy=c2c1
K_counter=k1
# X=np.concatenate((c2c1,X),axis=1)
for i, (mean, cov, color) in enumerate(zip(clf_CH.means_, clf_CH.covariances_,
                                           color_iter)):
    v, w = linalg.eigh(cov)
    K_counter=K_counter+1
    print(K_counter)
    if not np.any(Y_ == i):
        continue
    plt.scatter(X_xy[Y_ == i, 0], -X_xy[Y_ == i, 1], .8, color=color)

    # Plot an ellipse to show the Gaussian component
    angle = np.arctan2(w[1][0], w[0][0])
    angle = 180. * angle / np.pi  # convert to degrees
    v = 2. * np.sqrt(2.) * np.sqrt(v)
    ell = mpl.patches.Ellipse(mean, v[0], v[1],  angle, color=color)
    ell.set_clip_box(splot2.bbox)
    ell.set_alpha(.5)
    splot2.add_artist(ell)

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
# # X_xy=c2c1
# K_counter=0
# # X=np.concatenate((c2c1,X),axis=1)
# for i, (mean, cov, color) in enumerate(zip(clf_DB.means_, clf_DB.covariances_,
#                                            color_iter)):
#     v, w = linalg.eigh(cov)
#     K_counter=K_counter+1
#     print(K_counter)
#     if not np.any(Y_ == i):
#         continue
#     plt.scatter(X_xy[Y_ == i, 0], -X_xy[Y_ == i, 1], .8, color=color)
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