from open3d import *
import cv2 as cv
import open3d
import numpy as np
from sklearn import mixture
from sklearn.preprocessing import StandardScaler


class model():

    def __init__(self,image,depth):
        """

        :param image:
        :param depth:
        """
        self.image=image
        self.depth=depth

    def model_1(self):
        """
        Gaussian model (K=1) for each contour, OpenCV contour, open3d, normalization, no PCA, 3 features (x,y,z)
        it can detect which object is missing at the same frame
        it can <mostly> detect which object among first 7 part frame is assembled with witch one at second 6 part image by seeing
        that the kl_divergance of which detection is increased in comparision with the time there is no assembly
        (<mostly> because e.g except for little cubes with the larges part,...)
        """
        print(__doc__)
        # self.image = np.load("image_contours_4.npy")
        # self.depth = np.load("self.depth_contours_4.npy")
        imgray = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        kernel = np.ones((16, 16), np.uint8)
        imgray = cv.erode(imgray, kernel, iterations=1)
        ret, thresh = cv.threshold(imgray, 127, 255, 0)
        while 1:
            cv.imshow('thresh', thresh)
            key = cv.waitKey(1)
            if key == 27:
                break
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE,
                                              cv.CHAIN_APPROX_NONE)  # contours contains here the pixel index of the corresponding points
        contours_mean = np.empty([len(contours), 2])
        for n_contour in range(0, len(contours)):
            # find the mean value for the detected contours
            contours_mean[n_contour][:] = (contours[n_contour].mean(axis=0))
        contours_mean = np.around(contours_mean, decimals=0).astype(int)
        self.image_copy = np.copy(self.image)
        self.depth_copy = np.copy(self.depth)
        part_mean = np.empty([contours_mean.shape[0] - 1, 3])  # we cancel out the big frame contour
        part_covariance = np.empty([contours_mean.shape[0] - 1, 3, 3])


        for n_contour in range(1, len(contours)):  # remove out first contour
            self.image = np.copy(self.image_copy)
            self.depth = np.copy(self.depth_copy)
            self.image_contour_center = np.copy(self.image_copy)
            self.image_contour_center.fill(255.)  # here we create a white self.image
            self.image_contour_center[contours_mean[n_contour][1]][contours_mean[n_contour][0]] = self.image[contours_mean[n_contour][1]][
                contours_mean[n_contour][0]] # here we put non white color on the mean point of contour
            self.image_contour_center_3d = open3d.Image(self.image_contour_center)
            self.depth_3d = open3d.Image(self.depth)
            rgbd_image = open3d.create_rgbd_image_from_color_and_depth(self.image_contour_center_3d, self.depth_3d,
                                                                       convert_rgb_to_intensity=False)
            fx = 594.21
            fy = 591.04
            a = -0.0030711
            b = 3.3309495
            cx = 339.5
            cy = 242.7
            intrinsic = open3d.PinholeCameraIntrinsic(self.depth.shape[1], self.depth.shape[0], fx, fy, cx, cy)
            pcd_contour_mean = open3d.create_point_cloud_from_rgbd_image(rgbd_image, intrinsic)
            pcd_contour_mean.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

            for y in range(0, self.image.shape[0]):
                for x in range(0, self.image.shape[1]):
                    if cv.pointPolygonTest(contours[n_contour], (x, y), False) < 1:
                        self.image[y, x, :] = 100.
                        self.depth[y, x] = 100
            self.image_contours = np.copy(self.image)
            cv.drawContours(self.image_contours, contours[n_contour], -1, (0, 255, 0), 3)
            while 1:
                k = n_contour
                cv.imshow('Contour(object): k=%i' % k, self.image_contours)
                key = cv.waitKey(1)
                if key == 27:
                    break

            self.image_3d = open3d.Image(self.image)
            self.depth_3d = open3d.Image(self.depth)
            rgbd_image = open3d.create_rgbd_image_from_color_and_depth(self.image_3d, self.depth_3d, convert_rgb_to_intensity=False)
            intrinsic = open3d.PinholeCameraIntrinsic(self.depth.shape[1], self.depth.shape[0], fx, fy, cx, cy)
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

    def model_2(self,features):
        """
        GMM with K=number of detected closed contours by opencv, OpenCV contour, open3d, normalization, no PCA, 3 features (x,y,z)
        it can detect which object is missing at the same frame



        """
        print(__doc__)
        # self.image = np.load("image_contours_4.npy")
        # self.depth = np.load("self.depth_contours_4.npy")
        imgray = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        kernel = np.ones((16, 16), np.uint8)
        imgray = cv.erode(imgray, kernel, iterations=1)
        ret, thresh = cv.threshold(imgray, 127, 255, 0)
        while 1:
            cv.imshow('thresh', thresh)
            key = cv.waitKey(1)
            if key == 27:
                break
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE,
                                              cv.CHAIN_APPROX_NONE)  # contours contains here the pixel index of the corresponding points
        contours_mean = np.empty([len(contours), 2])
        for n_contour in range(0, len(contours)):
            # find the mean value for the detected contours
            contours_mean[n_contour][:] = (contours[n_contour].mean(axis=0))
        contours_mean = np.around(contours_mean, decimals=0).astype(int)
        self.image_copy = np.copy(self.image)
        self.depth_copy = np.copy(self.depth)
        part_mean = np.empty([contours_mean.shape[0] - 1, len(features)])  # we cancel out the big frame contour
        part_covariance = np.empty([contours_mean.shape[0] - 1, len(features), len(features)])

        self.image = np.copy(self.image_copy)
        self.depth = np.copy(self.depth_copy)
        self.image_contour_center = np.copy(self.image_copy)
        self.image_contour_center.fill(255.)  # here we create a white self.image
        for n_contour in range(1, len(contours)):

            self.image_contour_center[contours_mean[n_contour][1]][contours_mean[n_contour][0]] = \
                self.image[contours_mean[n_contour][1]][contours_mean[n_contour][0]]  # here we put non white color on the mean point of contour






        self.image_contour_center_3d = open3d.Image(self.image_contour_center)
        self.depth_3d = open3d.Image(self.depth)
        rgbd_image = open3d.create_rgbd_image_from_color_and_depth(self.image_contour_center_3d, self.depth_3d,
                                                                   convert_rgb_to_intensity=False)
        fx = 594.21
        fy = 591.04
        a = -0.0030711
        b = 3.3309495
        cx = 339.5
        cy = 242.7
        intrinsic = open3d.PinholeCameraIntrinsic(self.depth.shape[1], self.depth.shape[0], fx, fy, cx, cy)
        pcd_contour_mean = open3d.create_point_cloud_from_rgbd_image(rgbd_image, intrinsic)
        pcd_contour_mean.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        image_help=np.copy(self.image)
        depth_help=np.copy(self.depth)
        self.image.fill(100.)
        self.depth.fill(100)
        for n_contour in range(1, len(contours)):
            for y in range(0, self.image.shape[0]):
                for x in range(0, self.image.shape[1]):
                    if cv.pointPolygonTest(contours[n_contour], (x, y), False) > 0:
                        self.image[y, x, :] = np.array([255,0,0])
                        self.depth[y, x] = depth_help[y, x]

        self.image_contours = np.copy(self.image)
        cv.drawContours(self.image_contours, contours[1:], -1, (0, 255, 0), 3)
        while 1:

            cv.imshow('Contour(object): K=%i' %(len(contours)-1), self.image_contours)
            key = cv.waitKey(1)
            if key == 27:
                break

        self.image_3d = open3d.Image(self.image)
        self.depth_3d = open3d.Image(self.depth)
        rgbd_image = open3d.create_rgbd_image_from_color_and_depth(self.image_3d, self.depth_3d, convert_rgb_to_intensity=False)
        intrinsic = open3d.PinholeCameraIntrinsic(self.depth.shape[1], self.depth.shape[0], fx, fy, cx, cy)
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
        X_init=X[np.argsort(np.asarray(pcd_contour_mean.colors), axis=0)[:len(contours)-1, 0]] #this is for the center of all detected contours

        # End  Standardize features==============================================================================

        k1 = len(contours)-1
        X = X[X[:, 2] < 0.85, :]  # this is used to remove out the background witch is a way higher
        cluster_span = 1
        n_components_range = range(k1, k1 + cluster_span)
        cv_types = ['full']
        X_copy = np.copy(X)
        X = X[:, features]
        for cv_type in cv_types:
            for n_components in n_components_range:
                # Fit a Gaussian mixture with EM
                gmm = mixture.GaussianMixture(n_components=n_components,
                                              covariance_type=cv_type,
                                              means_init=X_init[:,features])
        gmm.fit(X)
        print("GMM number of gaussians(k)= ", n_components)
        part_mean = gmm.means_
        part_covariance = gmm.covariances_
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
