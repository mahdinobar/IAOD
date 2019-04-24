"""
This is a copy of GMM_11.py but with adding a method to show us the point cloud of the merged objects.
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
# from freenect import sync_get_depth, sync_get_video, init, close_device, open_device, set_led
from freenect import open_device, close_device, init

image=np.load('GMM_10_midpresentation_image_1.npy')
depth=np.load('GMM_10_midpresentation_depth_1.npy')

while 1:
    cv.imshow('image', image)
    key = cv.waitKey(1)
    if key==27:
        break

image_2=np.load('GMM_10_midpresentation_image_2.npy')
depth_2=np.load('GMM_10_midpresentation_depth_2.npy')
while 1:
    cv.imshow('image_2', image_2)
    key = cv.waitKey(1)
    if key==27:
        break

image_3=np.load('GMM_10_midpresentation_image_3.npy')
depth_3=np.load('GMM_10_midpresentation_depth_3.npy')
while 1:
    cv.imshow('image_3', image_3)
    key = cv.waitKey(1)
    if key==27:
        break

def GMM_separate(image,depth):
    """
    This function applies GMM with k=1 to each contours by considering its overall position (x,y,z) in original input
     frame then it returns the mean and covariance of gaussian for all contours in the input frame
    """
    print(__doc__)
    image_copy_3=np.copy(image)
    image_copy_4 = np.copy(image)
    imgray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    kernel = np.ones((16, 16), np.uint8)
    ret, thresh = cv.threshold(imgray, 127, 255, 0)
    while 1:
        cv.imshow('thresh before erosion', thresh)
        key = cv.waitKey(1)
        if key == 27:
            break
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE,
                                          cv.CHAIN_APPROX_NONE)
    cv.drawContours(image_copy_3, contours[1:], -1, (0, 255, 0), 3)
    while 1:
        cv.imshow('Image with contours before erosion', image_copy_3)
        key = cv.waitKey(1)
        if key == 27:
            break
    imgray = cv.erode(imgray, kernel, iterations=1)
    ret, thresh = cv.threshold(imgray, 127, 255, 0)
    while 1:
        cv.imshow('thresh after erosion', thresh)
        key = cv.waitKey(1)
        if key == 27:
            break
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE,
                                          cv.CHAIN_APPROX_NONE)  # contours contains here the pixel index of the corresponding points
    cv.drawContours(image_copy_4, contours[1:], -1, (0, 255, 0), 3)
    while 1:
        cv.imshow('Image with contours after erosion', image_copy_4)
        key = cv.waitKey(1)
        if key == 27:
            break
    contours_mean = np.empty([len(contours), 2])
    for n_contour in range(0, len(contours)):
        # find the mean value for the detected contours
        contours_mean[n_contour][:] = (contours[n_contour].mean(axis=0))
    contours_mean = np.around(contours_mean, decimals=0).astype(int)
    image_copy = np.copy(image)
    depth_copy = np.copy(depth)
    part_mean = np.empty([contours_mean.shape[0] - 1, 3])  # we cancel out the big frame contour
    part_covariance = np.empty([contours_mean.shape[0] - 1, 3, 3])


    for n_contour in range(1, len(contours)):  # remove out first contour
        image = np.copy(image_copy)
        depth = np.copy(depth_copy)
        image_contour_center = np.copy(image_copy)
        image_contour_center.fill(255.)  # here we create a white image
        image_contour_center[contours_mean[n_contour][1]][contours_mean[n_contour][0]] = image[contours_mean[n_contour][1]][
            contours_mean[n_contour][0]] # here we put non white color on the mean point of contour
        image_contour_center_3d = open3d.Image(image_contour_center)
        depth_3d = open3d.Image(depth)
        rgbd_image = open3d.create_rgbd_image_from_color_and_depth(image_contour_center_3d, depth_3d,
                                                                   convert_rgb_to_intensity=False)
        fx = 594.21
        fy = 591.04
        a = -0.0030711
        b = 3.3309495
        cx = 339.5
        cy = 242.7
        intrinsic = open3d.PinholeCameraIntrinsic(depth.shape[1], depth.shape[0], fx, fy, cx, cy)
        pcd_contour_mean = open3d.create_point_cloud_from_rgbd_image(rgbd_image, intrinsic)
        pcd_contour_mean.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        for y in range(0, image.shape[0]):
            for x in range(0, image.shape[1]):
                if cv.pointPolygonTest(contours[n_contour], (x, y), False) < 1:
                    image[y, x, :] = 100.
                    depth[y, x] = 100
        image_contours = np.copy(image)
        cv.drawContours(image_contours, contours[n_contour], -1, (0, 255, 0), 3)
        while 1:
            k = n_contour
            cv.imshow('Contour(object): k=%i' % k, image_contours)
            key = cv.waitKey(1)
            if key == 27:
                break

        image_3d = open3d.Image(cv.cvtColor(image, cv.COLOR_RGB2BGR))
        depth_3d = open3d.Image(depth)
        rgbd_image = open3d.create_rgbd_image_from_color_and_depth(image_3d, depth_3d, convert_rgb_to_intensity=False)
        intrinsic = open3d.PinholeCameraIntrinsic(depth.shape[1], depth.shape[0], fx, fy, cx, cy)
        pcd = open3d.create_point_cloud_from_rgbd_image(rgbd_image, intrinsic)
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        open3d.draw_geometries([pcd])
        pcd_points = np.asarray(pcd.points)
        pcd_colours = np.asarray(pcd.colors)
        X = np.hstack((pcd_points, pcd_colours))

        # Start Standardize features==============================================================================
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)  # here X_init has the big frame also
        X_init = X[np.asarray(pcd_contour_mean.colors).argsort(axis=0)[0, 0]]
        # End  Standardize features==============================================================================

        k1 = 1
        X = X[X[:, 2] < 0.14, :]  # this is used to remove out the background witch is a way higher
        cluster_span = 1
        n_components_range = range(k1, k1 + cluster_span)
        cv_types = ['full']
        X_copy = np.copy(X)
        X = X[:, 0:3]
        for cv_type in cv_types:
            for n_components in n_components_range:
                # Fit a Gaussian mixture with EM
                gmm = mixture.GaussianMixture(n_components=n_components,
                                              covariance_type=cv_type,
                                              means_init=np.reshape(X_init[0:3],(1,3)))
                gmm.fit(X)
                print("GMM number of gaussians(k)= ", n_components)
        part_mean[n_contour - 1, :] = (gmm.means_)
        part_covariance[n_contour - 1, :] = (gmm.covariances_)
        Y_ = gmm.predict(X)
        color = np.array(
            [[204, 0, 0], [0, 204, 0], [0, 0, 204], [255, 0, 127], [255, 255, 0], [127, 0, 255], [255, 128, 0],
             [102, 51, 0], [255, 153, 153], [153, 255, 255], [0, 102, 102]]) / 255
        for i in range(np.unique(Y_).min(), np.unique(Y_).max() + 1):
            # plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)
            X_copy[Y_ == i, 3:6] = color[i]
        pcd.colors = Vector3dVector(X_copy[:, 3:6])
        pcd.points = Vector3dVector(X_copy[:, 0:3])
        open3d.draw_geometries([pcd])

    return part_mean, part_covariance

def merged_point_cloud(image,depth,index):
    """
    """
    print(__doc__)
    imgray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    kernel = np.ones((16, 16), np.uint8)
    ret, thresh = cv.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE,
                                          cv.CHAIN_APPROX_NONE)
    cv.drawContours(image, contours[index], -1, (0, 255, 0), 3)
    for y in range(0, image.shape[0]):
        for x in range(0, image.shape[1]):
            if cv.pointPolygonTest(contours[index], (x, y), False) > 0:
                image[y, x, :] = np.array([0,255,0])

    while 1:
        cv.imshow('Image with contours before erosion', image)
        key = cv.waitKey(1)
        if key == 27:
            break
    fx = 594.21
    fy = 591.04
    a = -0.0030711
    b = 3.3309495
    cx = 339.5
    cy = 242.7
    image_3d = open3d.Image(cv.cvtColor(image, cv.COLOR_RGB2BGR))
    depth_3d = open3d.Image(depth)
    rgbd_image = open3d.create_rgbd_image_from_color_and_depth(image_3d, depth_3d, convert_rgb_to_intensity=False)
    intrinsic = open3d.PinholeCameraIntrinsic(depth.shape[1], depth.shape[0], fx, fy, cx, cy)
    pcd = open3d.create_point_cloud_from_rgbd_image(rgbd_image, intrinsic)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    open3d.draw_geometries([pcd])




# part_mean, part_covariance=GMM_separate(image,depth)
# part_mean_2,part_covariance_2=GMM_separate(image_3,depth_3)
#
# covariance_distance=np.empty([part_covariance.shape[0],part_covariance_2.shape[0]])
# mean_distance=np.empty_like(covariance_distance)
# kl_div=np.empty_like(covariance_distance)
# Hellinger=np.empty_like(covariance_distance)
# for c1 in range (0,part_covariance.shape[0]):
#     for c2 in range (0,part_covariance_2.shape[0]):
#         covariance_distance[c1,c2]=(np.linalg.norm(part_covariance_2[c2]-part_covariance[c1]))
#         mean_distance[c1, c2] = (np.linalg.norm(part_mean_2[c2] - part_mean[c1]))
#         kl_div[c1,c2]=0.5*(np.log(np.linalg.det(part_covariance[c1])/np.linalg.det(part_covariance_2[c2]))
#                            -part_mean.shape[1]
#                            +np.trace(np.matmul(np.linalg.inv(part_covariance[c1]),part_covariance_2[c2]))
#                            +np.matmul(np.matmul((part_mean[c1]-part_mean_2[c2]),np.linalg.inv(part_covariance[c1]))
#                                       ,(part_mean[c1]-part_mean_2[c2])))
#         Hellinger[c1,c2]=math.sqrt(1-(np.linalg.det(part_covariance_2[c2])*np.linalg.det(part_covariance[c1]))**(1/4)
#                                    /np.linalg.det(part_covariance_2[c2]/2+part_covariance[c1]/2)**(1/2)
#                                    *math.exp(-np.matmul(np.matmul((part_mean[c1]-part_mean_2[c2]),
#                                                                   np.linalg.inv(part_covariance_2[c2]/2+part_covariance[c1]/2))
#                                                         ,(part_mean[c1]-part_mean_2[c2]))/8))
# kl_argsort=np.argsort(kl_div,axis=0)#to see if the result is correct
# index=merged_component=np.argmax(np.sort(kl_div,axis=0)[0,:],axis=0)#this gives the contour number of the merged object at assembled frame
merged_point_cloud(image_3,depth_3,1)

print('end')