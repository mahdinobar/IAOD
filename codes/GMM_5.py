"""
GMM for CH best parts, K=7, with initial condition without background,3 features, remove background, no PCA:
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


print(__doc__)
# Start cropping=======================================================================================

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
	if event == cv.EVENT_LBUTTONDOWN:
		ref_point = [(x, y)]
		cropping = True

	# check to see if the left mouse button was released
	elif event == cv.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		ref_point.append((x, y))
		cropping = False

		# draw a rectangle around the region of interest
		cv.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
		cv.imshow("image", image)

def get_video():
    array, _ = freenect.sync_get_video()
    array = cv.cvtColor(array, cv.COLOR_RGB2BGR)
    return array

image=get_video()

clone = image.copy()
cv.namedWindow("image")
cv.setMouseCallback("image", shape_selection)

# keep looping until the 'q' key is pressed
while True:
	# display the image and wait for a keypress
	cv.imshow("image", image)
	key = cv.waitKey(1) & 0xFF

	# if the 'r' key is pressed, reset the cropping region
	if key == ord("r"):
		image = clone.copy()

	# if the 'c' key is pressed, break from the loop
	elif key == ord("c"):
		break

if len(ref_point) == 2:
	crop_img = clone[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
	cv.imshow("crop_img", crop_img)
	cv.waitKey(0)

cv.destroyAllWindows()

def get_cropped_video():
    array, _ = freenect.sync_get_video()
    array=array[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
    array = array.astype(np.uint8)
    return array

def get_cropped_depth():
    array, _ = freenect.sync_get_depth(format=freenect.DEPTH_REGISTERED)
    array = array[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
    # print (array.shape)
    array = array.astype(np.uint8)
    return array
# Finish cropping=======================================================================================


# image = get_cropped_video()
# depth=get_cropped_depth()
# np.save("image_contours_4.npy",image)
# np.save("depth_contours_4.npy",depth)
image =np.load("image_contours_4.npy")
depth=np.load("depth_contours_4.npy")
imgray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
kernel = np.ones((15,15),np.uint8)
imgray = cv.erode(imgray,kernel,iterations = 1)
ret, thresh = cv.threshold(imgray, 127, 255, 0)
while 1:
    cv.imshow('thresh', thresh)
    key = cv.waitKey(1)
    if key==27:
        break
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE) #contours contains here the pixel index of the corresponding points
image_contours=np.copy(image)
cv.drawContours(image_contours, contours[1:], -1, (0,255,0), 3)
#find the mean value for the detected contours
contours_mean=np.empty([len(contours)-1,2])
for n_contour in range(0,len(contours)-1):
    contours_mean[n_contour][:]=(contours[n_contour+1].mean(axis=0))
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
contours_mean_pcd_xyz=np.asarray(pcd_contour_mean.points)[np.asarray(pcd_contour_mean.colors).argsort(axis=0)[0:contours_mean.shape[0],0]]
contours_mean_pcd_rgb=np.asarray(pcd_contour_mean.colors)[np.asarray(pcd_contour_mean.colors).argsort(axis=0)[0:contours_mean.shape[0],0]]

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
# X_init=np.hstack((np.asarray(pcd.points)[np.asarray(pcd_contour_mean.colors).argsort(axis=0)[0:contours_mean.shape[0],0]],
#                   np.asarray(pcd.colors)[np.asarray(pcd_contour_mean.colors).argsort(axis=0)[0:contours_mean.shape[0],0]]))

# np.save("raw_DATA_last_run_GMM_3.npy",X)
# X=np.load("DATA_11march.npy")




# #Start Standardize features==============================================================================
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler.fit(X)
# X=scaler.transform(X)
# # End  Standardize features==============================================================================

#
# Generate random sample, two components
np.random.seed(0)

# #Start background detection with GMM--------------------------------------------------------------------------------
# from sklearn import metrics
# import numpy as np
# import itertools
# from scipy import linalg
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# from sklearn import mixture
# import freenect
# import cv2
#
# lowest_bic = np.infty
# bic = []
# lowest_SC = np.infty
# lowest_CH = np.infty
# lowest_DB = np.infty
# SC=[]
# CH=[]
#
# n_components_range = range(2, 3)
# cv_types = ['full']
# for cv_type in cv_types:
#     for n_components in n_components_range:
#         # Fit a Gaussian mixture with EM
#         gmm = mixture.GaussianMixture(n_components=n_components,
#                                       covariance_type=cv_type)
#         gmm.fit(X)
#         print("backgroundremoval GMM number of gaussians(k)= ", n_components)
#
#         CH.append(metrics.calinski_harabaz_score(X, gmm.predict(X)))
#         if CH[-1] < lowest_CH:
#             lowest_CH = CH[-1]
#             best_gmm_CH = gmm
#
# CH=np.array(CH)
# color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
#                               'darkorange','red','green','blue','yellow','cyan','black'])
#
# clf_CH=best_gmm_CH
# #End background detection with GMM--------------------------------------------------------------------------------
#
# #Start background removal--------------------------------------------------------------------------------
# Y_ = clf_CH.predict(X)
# if X[Y_==0,2].mean()>X[Y_==1,2].mean():
#     X = X[Y_ == 0, :]
#     X_copy=X_copy[Y_ == 0, :]
# else:
#     X = X[Y_ == 1, :]
#     X_copy = X_copy[Y_ == 1, :]
# #End background removal--------------------------------------------------------------------------------

#Start Standardize features==============================================================================
scaler = StandardScaler()
scaler.fit(X)
X=scaler.transform(X)
X_init=X[np.asarray(pcd_contour_mean.colors).argsort(axis=0)[0:contours_mean.shape[0],0]]
# End  Standardize features==============================================================================


def remove_background(data):
    gmm = mixture.GaussianMixture(n_components=2,
                                  covariance_type="spherical")
    labels = gmm.fit_predict(data)
    average_z = np.zeros((2,1))
    for i in range(0,np.max(labels)+1):
        average_z[i] = np.average(data[np.where(labels == i), 2])
    print(average_z)
    background_label= np.argmin(average_z)
    foreground = np.delete(data,np.where(labels ==background_label),axis=0)
    return foreground


lowest_bic = np.infty
bic = []
lowest_SC = np.infty
lowest_CH = np.infty
lowest_DB = np.infty
SC=[]
CH=[]
k1=5
cluster_span=3
n_components_range = range(k1, k1+cluster_span)
cv_types = ['full']
X=remove_background(X)
X_copy=np.copy(X)
X=X[:,0:3]
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type=cv_type,
                                      means_init=X_init[:n_components,0:3])
        gmm.fit(X)
        print("GMM number of gaussians(k)= ",n_components)
        CH.append(metrics.calinski_harabaz_score(X, gmm.predict(X)))
        if CH[-1] < lowest_CH:
            lowest_CH = CH[-1]
            best_gmm_CH = gmm

CH=np.array(CH)
color_iter = itertools.cycle(['blue','yellow','navy', 'turquoise', 'cornflowerblue',
                              'darkorange','red','green','cyan','black'])

clf_CH=best_gmm_CH

#End background removal----------------------------------------------------------------------------------

bars = []



bars_CH = []
# Plot the CH scores-------------------------------------------------------------------------
spl = plt.subplot(2, 1, 1)
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos_CH = np.array(n_components_range) + .2 * (i - 2)
    bars_CH.append(plt.bar(xpos_CH, CH[i * len(n_components_range):
                                  (i + 1) * len(n_components_range)],
                        width=.2, color=color))
plt.xticks(n_components_range)
plt.ylim([CH.min() * 1.01 - .01 * CH.max(), CH.max()])
plt.title('CH score per model')
spl.set_xlabel('Number of components')
spl.legend([b[0] for b in bars_CH], cv_types)

#End plot CH scores--------------------------------------------------------------------------

#Start Plot the CH winner-------------------------------------------------------------------------
splot2 = plt.subplot(2, 1, 2)
Y_ = clf_CH.predict(X)
# X_copy=c2c1
K_counter=0
for i, (mean, cov, color) in enumerate(zip(clf_CH.means_, clf_CH.covariances_,
                                           color_iter)):
    v, w = linalg.eigh(cov)
    K_counter=K_counter+1

    if not np.any(Y_ == i):
        continue
    plt.scatter(X[Y_ == i, 0], -X[Y_ == i, 1], .8, color=color)

    # Plot an ellipse to show the Gaussian component
    angle = np.arctan2(w[1][0], w[0][0])
    angle = 180. * angle / np.pi  # convert to degrees
    v = 2. * np.sqrt(2.) * np.sqrt(v)
    ell = mpl.patches.Ellipse(mean, v[0], v[1],  angle, color=color)
    ell.set_clip_box(splot2.bbox)
    ell.set_alpha(.5)
    splot2.add_artist(ell)
print("The optimal number of clusters= ", K_counter)
plt.xticks(())
plt.yticks(())
plt.title('Selected GMM (CH winner): * model, * components')
plt.subplots_adjust(hspace=.35, bottom=.02)
plt.show()
print('END GMM_1')
#End Plot the CH winner-------------------------------------------------------------------------


color=np.array([[204,0,0],[0,204,0],[0,0,204],[255,0,127],[255,255,0],[127,0,255],[255,128,0],[102,51,0],[255,153,153],[153,255,255],[0,102,102]])/255
for i in range (np.unique(Y_).min(),np.unique(Y_).max()+1):
    # plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)
    X_copy[Y_ == i, 3:6]=color[i]

pcd.colors = Vector3dVector(X_copy[:, 3:6])
pcd.points = Vector3dVector(X_copy[:, 0:3])
open3d.draw_geometries([pcd])