# IAOD

EPFL Semester Project: Inferring Assembly Objects from Demonstration

Author: Mahdi Nobar

Supervisor: Prof Aude Billard

Assistant: Dr Athanasios Polydoros

Initially, various unsupervised learning algorithms including GMM, Birch, Mean-Shift, K-means and DBSCAN were used to cluster the image and depth sensor data. However, it was not possible to fit the clusters on each objects quite separately in as much as data are too noisy and the components are too small and near each other. Afterwards in this project, the capability of inferring small assembly objects is demonstrated by combining the machine learning unsupervised learning with computer vision algorithms. It is proved that vision techniques can assign a contour around each object. Next, the object type is inferred by fitting a Gaussian model to each detected contours. Then, the noise effect on the results decreased considerably by utilizing image registration. Finally after reducing the noise effect, it was possible to infer which pair of the objects merged together. Nonetheless, the hyper-parameters of the final proposed algorithm require to be tuned for any new scenario; These hyper-parameters include threshold for removal of the contours detected for the holes inside the hollowed gear, the voxel size for down-sampling, the ratio of maximum distance between point clouds used for noise reduction, and the point numbers for K-Nearest Neighbors used for FPFH feature calculation.
