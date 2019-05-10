from open3d import *
import cv2 as cv
import open3d
import numpy as np
from sklearn import mixture
from sklearn.preprocessing import StandardScaler
from utils import point_cloud, point_cloud_with_registration
from sklearn.decomposition import PCA


class model():

    def __init__(self,image_source,depth_source,image_target=None,depth_target=None):
        """

        :param image:
        :param depth:
        """
        self.image=image_source
        self.depth=depth_source
        self.image_target = image_target
        self.depth_target = depth_target

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
            pcd_contour_mean=point_cloud(self.image_contour_center,self.depth)
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
            pcd=point_cloud(self.image,self.depth)
            # open3d.draw_geometries([pcd])
            # reg=open3d.registration.registration_icp(pcd,pcd,0.001)
            # fpfh=open3d.registration.compute_fpfh_feature(pcd, KDTreeSearchParamHybrid(radius = 0.1, max_nn = 100))
            pcd_points = np.asarray(pcd.points)
            pcd_colours = np.asarray(pcd.colors)
            X = np.hstack((pcd_points, pcd_colours))

            #standardize features
            scaler = StandardScaler()
            scaler.fit(X)
            X = scaler.transform(X)  # here X_init has the big frame also
            X_init = X[np.asarray(pcd_contour_mean.colors).argsort(axis=0)[0, 0]]


            #downsampling the point cloud to pick the average value inside each voxel
            downpcd = open3d.voxel_down_sample(pcd, voxel_size=0.0000001)
            pcd_points = np.asarray(downpcd.points)
            pcd_colours = np.asarray(downpcd.colors)
            X = np.hstack((pcd_points, pcd_colours))
            scaler = StandardScaler()
            scaler.fit(X)
            X = scaler.transform(X)
            k1 = 1
            X = X[X[:, 2] < -0.0004, :]  #this is used to remove out the background witch is a way higher it needs to be generalized for any taken frame
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
        GMM with K=number of detected closed contours by opencv, OpenCV contour, open3d, normalization, no PCA, 4 features (x,y,z,RED)
        removed background
        removed the object shade by putting red color ar inside each contour
        with cluster mean initialization

        This now is correct only for features =[0,1,2,3]


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

        X = X[:, features]
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
        X_copy1 = np.copy(X)
        X_copy=np.hstack((X_copy1,np.empty((X_copy1.shape[0],2))))
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

    def model_3(self,num_features=33):
        """
        same as model_1 but gaussians are trained with FPFH features(33-dimensional) of each point
        First I created model_3 based on model_1, then I use point cloud of image and depth data to
         calculate the surface normals and then I compute FPFH of the point cloud which has same
         number of points but it has 33 features. The I use FPFH features to train each gaussians.
          Finally after tuning the parameters of FPFH at the best result it can only match the
          objects with each other but it cannot be used to distinguish which object has been merged
          with which one. Also, the problem that if we take two frames same frames with exactly same
           objects (noise effect) still exists and if it is solved it might help us to have the results
            that we need. I also tried to apply PCA and change number of features of FPFH for training
            gaussians and training with and without providing initial conditions for gaussian mean, but
             none of them helps us.
        """
        # self.image = np.load("image_contours_4.npy")
        # self.depth = np.load("self.depth_contours_4.npy")
        imgray = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        kernel = np.ones((16, 16), np.uint8)
        imgray = cv.erode(imgray, kernel, iterations=1)
        ret, thresh = cv.threshold(imgray, 127, 255, 0)
        # while 1:
        #     cv.imshow('thresh', thresh)
        #     key = cv.waitKey(1)
        #     if key == 27:
        #         break
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE,
                                              cv.CHAIN_APPROX_NONE)  # contours contains here the pixel index of the corresponding points
        contours_mean = np.empty([len(contours), 2])
        for n_contour in range(0, len(contours)):
            # find the mean value for the detected contours
            contours_mean[n_contour][:] = (contours[n_contour].mean(axis=0))
        contours_mean = np.around(contours_mean, decimals=0).astype(int)
        self.image_copy = np.copy(self.image)
        self.depth_copy = np.copy(self.depth)
        part_mean = np.empty([contours_mean.shape[0] - 1, num_features])  # we cancel out the big frame contour
        part_covariance = np.empty([contours_mean.shape[0] - 1, num_features, num_features])


        for n_contour in range(1, len(contours)):  # remove out first contour
            self.image = np.copy(self.image_copy)
            self.depth = np.copy(self.depth_copy)
            self.image_contour_center = np.copy(self.image_copy)
            self.image_contour_center.fill(255.)  # here we create a white self.image
            self.image_contour_center[contours_mean[n_contour][1]][contours_mean[n_contour][0]] = self.image[contours_mean[n_contour][1]][
                contours_mean[n_contour][0]] # here we put non white color on the mean point of contour

            pcd_contour_mean=point_cloud(self.image_contour_center,self.depth)


            for y in range(0, self.image.shape[0]):
                for x in range(0, self.image.shape[1]):
                    if cv.pointPolygonTest(contours[n_contour], (x, y), False) < 1:
                        self.image[y, x, :] = 100.
                        self.depth[y, x] = 100.
            self.image_contours = np.copy(self.image)
            cv.drawContours(self.image_contours, contours[n_contour], -1, (0, 255, 0), 3)

            pcd=point_cloud(self.image,self.depth)
            voxel_size=0.05
            radius_normal=voxel_size*2
            open3d.geometry.estimate_normals(pcd, open3d.geometry.KDTreeSearchParamHybrid(
                radius=radius_normal, max_nn=13))
            radius_feature = voxel_size * 5
            pcd_fpfh = open3d.registration.compute_fpfh_feature(pcd,
                                            open3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

            X=pcd_fpfh.data.transpose()

            # Start Standardize features==============================================================================
            # scaler = StandardScaler()
            # scaler.fit(X)
            # X = scaler.transform(X)  # here X_init has the big frame also
            X_init = X[np.asarray(pcd_contour_mean.colors).argsort(axis=0)[0, 0]]
            # End  Standardize features==============================================================================

            k1 = 1
            pcd_points = np.asarray(pcd.points)
            pcd_colours = np.asarray(pcd.colors)
            X_old = np.hstack((pcd_points, pcd_colours))
            X = X[X_old[:, 2] < -0.00041, :]  # this is used to remove out the background witch is a way higher
            X_old = X_old[X_old[:, 2] < -0.00041, :]
            cluster_span = 1
            n_components_range = range(k1, k1 + cluster_span)
            cv_types = ['full']
            X = X[:, :num_features]
            for cv_type in cv_types:
                for n_components in n_components_range:
                    # Fit a Gaussian mixture with EM
                    gmm = mixture.GaussianMixture(n_components=n_components,
                                                  covariance_type=cv_type,
                                                  means_init=np.reshape(X_init[:num_features],(1,num_features)))
                    gmm.fit(X)
                    print("GMM number of gaussians(k)= ", n_components)
            part_mean[n_contour - 1, :] = (gmm.means_)
            part_covariance[n_contour - 1, :] = (gmm.covariances_)
            Y_ = gmm.predict(X)
            color = np.array(
                [[204, 0, 0], [0, 204, 0], [0, 0, 204], [255, 0, 127], [255, 255, 0], [127, 0, 255], [255, 128, 0],
                 [102, 51, 0], [255, 153, 153], [153, 255, 255], [0, 102, 102]]) / 255
            for i in range(np.unique(Y_).min(), np.unique(Y_).max() + 1):
                X_old[Y_ == i, 3:6] = color[i]
            pcd.colors = Vector3dVector(X_old[:, 3:6])
            pcd.points = Vector3dVector(X_old[:, 0:3])
            open3d.draw_geometries([pcd])
        return part_mean, part_covariance

    def model_4(self):
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
        self.image_target_copy = np.copy(self.image_target)
        self.depth_target_copy = np.copy(self.depth_target)
        part_mean = np.empty([contours_mean.shape[0] - 1, 3])  # we cancel out the big frame contour
        part_covariance = np.empty([contours_mean.shape[0] - 1, 3, 3])

        for n_contour in range(1, len(contours)):  # remove out first contour
            self.image = np.copy(self.image_copy)
            self.depth = np.copy(self.depth_copy)
            self.image_target = np.copy(self.image_target_copy)
            self.depth_target = np.copy(self.depth_target_copy)
            self.image_contour_center = np.copy(self.image_copy)
            self.image_contour_center.fill(255.)  # here we create a white self.image
            self.image_contour_center[contours_mean[n_contour][1]][contours_mean[n_contour][0]] = self.image[contours_mean[n_contour][1]][
                contours_mean[n_contour][0]] # here we put non white color on the mean point of contour
            pcd_contour_mean=point_cloud(self.image_contour_center,self.depth)
            for y in range(0, self.image.shape[0]):
                for x in range(0, self.image.shape[1]):
                    if cv.pointPolygonTest(contours[n_contour], (x, y), False) < 1:
                        self.image[y, x, :] = 100.
                        self.depth[y, x] = 0
                        self.image_target[y, x, :] = 100.
                        self.depth_target[y, x] = 0
            self.image_contours = np.copy(self.image)
            cv.drawContours(self.image_contours, contours[n_contour], -1, (0, 255, 0), 3)
            while 1:
                k = n_contour
                cv.imshow('Contour(object): k=%i' % k, self.image_contours)
                key = cv.waitKey(1)
                if key == 27:
                    break
            pcd=point_cloud_with_registration(self.image,self.depth,self.image_target,self.depth_target,ratio=1)
            # open3d.draw_geometries([pcd])
            # reg=open3d.registration.registration_icp(pcd,pcd,0.001)
            # fpfh=open3d.registration.compute_fpfh_feature(pcd, KDTreeSearchParamHybrid(radius = 0.1, max_nn = 100))
            pcd_points = np.asarray(pcd.points)
            pcd_colours = np.asarray(pcd.colors)
            X = np.hstack((pcd_points, pcd_colours))

            #standardize features
            scaler = StandardScaler()
            scaler.fit(X)
            X = scaler.transform(X)  # here X_init has the big frame also
            # X_init = X[np.asarray(pcd_contour_mean.colors).argsort(axis=0)[0, 0]]

            k1 = 1
            # X = X[X[:, 2] < 0.026, :]  #this is used to remove out the background witch is a way higher it needs to be generalized for any taken frame
            cluster_span = 1
            n_components_range = range(k1, k1 + cluster_span)
            cv_types = ['full']
            X_copy = np.copy(X)
            X = X[:, 0:3]
            for cv_type in cv_types:
                for n_components in n_components_range:
                    # Fit a Gaussian mixture with EM
                    gmm = mixture.GaussianMixture(n_components=n_components,
                                                  covariance_type=cv_type)
                                                  # means_init=np.reshape(X_init[0:3],(1,3)))
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
