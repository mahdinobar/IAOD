"""
GMM K=1for each contour, OpenCV contour, open3d, normalization, no PCA, 3 features (x,y,z)
it can detect which object is missing at the same frame
it can <mostly> detect which object among first 7 part frame is assembled with witch one at second 6 part image by seeing
that the kl_divergance of which detection is increased in comparision with the time there is no assembly
(<mostly> because e.g except for little cubes with the larges part,...)
"""

import cv2 as cv
import numpy as np
import math
from models import model
from Data import data
from measure_distance import models_distance

image,depth, cropped_window=data.get_data(None)
# np.save('GMM_10_midpresentation_image_1.npy', image)
# np.save('GMM_10_midpresentation_depth_1.npy', depth)
while 1:
    cv.imshow('image', image)
    key = cv.waitKey(1)
    if key==27:
        break

image_2,depth_2,_=data.get_data(cropped_window)
# np.save('GMM_10_midpresentation_image_2.npy', image_2)
# np.save('GMM_10_midpresentation_depth_2.npy', depth_2)
while 1:
    cv.imshow('image_2', image_2)
    key = cv.waitKey(1)
    if key==27:
        break

mean_set_1,cov_set_1 = model(image,depth).model_1()
mean_set_2,cov_set_2=model(image_2,depth_2).model_1()

covariance_distance=models_distance(mean_set_1,cov_set_1,mean_set_2,cov_set_2).covariance_distance()
mean_distance=models_distance(mean_set_1,cov_set_1,mean_set_2,cov_set_2).mean_distance()
KL_divergence=models_distance(mean_set_1,cov_set_1,mean_set_2,cov_set_2).KL_divergence()
Hellinger=models_distance(mean_set_1,cov_set_1,mean_set_2,cov_set_2).Hellinger()

kl_argsort=np.argsort(KL_divergence,axis=0)#to see if the result is correct

print('end')