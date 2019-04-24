"""

"""
import numpy as np
import cv2 as cv
from open3d import *
import freenect
import open3d
import numpy as np
import itertools
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import mixture
import freenect
import cv2

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


image = get_cropped_video()
depth=get_cropped_depth()


imgray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
kernel = np.ones((13,13),np.uint8)
imgray = cv.erode(imgray,kernel,iterations = 1)
ret, thresh = cv.threshold(imgray, 127, 255, 0)
while 1:
    cv.imshow('thresh', thresh)
    key = cv.waitKey(1)
    if key==27:
        break
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE) #contours contains here the pixel index of the corresponding points
image_contours=np.copy(image)
cv.drawContours(image_contours, contours, -1, (0,255,0), 3)
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
contours_mean_pcd_xyz=np.asarray(pcd_contour_mean.points)[np.asarray(pcd_contour_mean.colors).argsort(axis=0)[0:contours_mean.shape[0],0]]
contours_mean_pcd_rgb=np.asarray(pcd_contour_mean.colors)[np.asarray(pcd_contour_mean.colors).argsort(axis=0)[0:contours_mean.shape[0],0]]

open3d.draw_geometries([pcd_contour_mean])