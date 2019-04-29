"""

"""

import cv2 as cv
import numpy as np
from models import model
from Data import data
from measure_distance import models_distance

image,depth, cropped_window=data.get_data(None)
np.save('image_1.npy', image)
np.save('depth_1.npy', depth)
# image=np.load('image_1.npy')
# depth=np.load('depth_1.npy')
# image=np.load('GMM_10_midpresentation_image_1.npy')
# depth=np.load('GMM_10_midpresentation_depth_1.npy')

while 1:
    cv.imshow('image', image)
    key = cv.waitKey(1)
    if key==27:
        break

image_2,depth_2,_=data.get_data(cropped_window)
np.save('image_2.npy', image_2)
np.save('depth_2.npy', depth_2)
# image_2=np.load('image_2.npy')
# depth_2=np.load('depth_2.npy')
# image_2=np.load('GMM_10_midpresentation_image_3.npy')
# depth_2=np.load('GMM_10_midpresentation_depth_3.npy')

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