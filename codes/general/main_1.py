"""

"""

import cv2 as cv
import numpy as np
from models import model
from Data import data
from measure_distance import models_distance

# image_source,depth_source, cropped_window=data.get_data(None)
# np.save('image_7parts_1.npy', image_source)
# np.save('depth_7parts_1.npy', depth_source)
# image_source=np.load('image_7parts.npy')
# depth_source=np.load('depth_7parts.npy')
# image_source=np.load('image_1.npy')
# depth_source=np.load('depth_1.npy')
# image_source=np.load('GMM_10_midpresentation_image_1.npy')
# depth_source=np.load('GMM_10_midpresentation_depth_1.npy')
# image_source=np.load('image_samefirst.npy')
# depth_source=np.load('depth_samefirst.npy')
# image_source=np.load('image_samefirst_4 parts.npy')
# depth_source=np.load('depth_samefirst_4 parts.npy')
# image_target=np.load('image_samesecond_4 parts.npy')
# depth_target=np.load('depth_samesecond_4 parts.npy')
# image_source=np.load('image_7parts_1.npy')
# depth_source=np.load('depth_7parts_1.npy')
image_source=np.load('image_7parts_1.npy')
depth_source=np.load('depth_7parts_1.npy')
# image_source=np.load('image_Lshaped_position_1.npy')
# depth_source=np.load('depth_Lshaped_position_1.npy')
# image_source=np.load('image_smallobject_position_1.npy')
# depth_source=np.load('depth_smallobject_position_1.npy')
while 1:
    cv.imshow('image_source', image_source)
    key = cv.waitKey(1)
    if key==27:
        break
# image_2_source=image_source
# depth_2_source=depth_source
# image_2_source,depth_2_source,_=data.get_data(cropped_window)
# np.save('image_7parts_2.npy', image_2_source)
# np.save('depth_7parts_2.npy', depth_2_source)
# image_2_source=np.load('image_6parts.npy')
# depth_2_source=np.load('depth_6parts.npy')
# image_2_source=np.load('image_2_source.npy')
# depth_2_source=np.load('depth_2_source.npy')
# image_2_source=np.load('GMM_10_midpresentation_image_3.npy')
# depth_2_source=np.load('GMM_10_midpresentation_depth_3.npy')
# image_2_source=np.load('image_samesecond.npy')
# depth_2_source=np.load('depth_samesecond.npy')
# image_2_source=np.load('image_samefirst_4 parts.npy')
# depth_2_source=np.load('depth_samefirst_4 parts.npy')
# image_2_target=np.load('image_samesecond_4 parts.npy')
# depth_2_target=np.load('depth_samesecond_4 parts.npy')
image_2_source=np.load('image_7parts_assembledLandSmall_1.npy')
depth_2_source=np.load('depth_7parts_assembledLandSmall_1.npy')
# image_2_source=np.load('image_Lshaped_position_2.npy')
# depth_2_source=np.load('depth_Lshaped_position_2.npy')
# image_2_source=np.load('image_smallobject_position_2.npy')
# # depth_2_source=np.load('depth_smallobject_position_2.npy')
while 1:
    cv.imshow('image_2_source', image_2_source)
    key = cv.waitKey(1)
    if key==27:
        break

# image_target,depth_target,_=data.get_data(cropped_window)
# np.save('image_7parts_3.npy', image_target)
# np.save('depth_7parts_3.npy', depth_target)
# image_target=np.load('image_7parts_3.npy')
# depth_target=np.load('depth_7parts_3.npy')
image_target=np.load('image_7parts_2.npy')
depth_target=np.load('depth_7parts_2.npy')
while 1:
    cv.imshow('image_target', image_target)
    key = cv.waitKey(1)
    if key==27:
        break

# image_2_target,depth_2_target,_=data.get_data(cropped_window)
# np.save('image_7parts_4.npy', image_2_target)
# np.save('depth_7parts_4.npy', depth_2_target)
# image_2_target=np.load('image_7parts_4.npy')
# depth_2_target=np.load('depth_7parts_4.npy')
image_2_target=np.load('image_7parts_assembledLandSmall_2.npy')
depth_2_target=np.load('depth_7parts_assembledLandSmall_2.npy')
while 1:
    cv.imshow('image_2_target', image_2_target)
    key = cv.waitKey(1)
    if key==27:
        break


mean_set_1,cov_set_1 = model(image_source,depth_source,image_target,depth_target).model_4()
mean_set_2,cov_set_2=model(image_2_source,depth_2_source,image_2_target,depth_2_target).model_4()

covariance_distance=models_distance(mean_set_1,cov_set_1,mean_set_2,cov_set_2).covariance_distance()
mean_distance=models_distance(mean_set_1,cov_set_1,mean_set_2,cov_set_2).mean_distance()
KL_divergence=models_distance(mean_set_1,cov_set_1,mean_set_2,cov_set_2).KL_divergence()
Hellinger=models_distance(mean_set_1,cov_set_1,mean_set_2,cov_set_2).Hellinger()

kl_argsort=np.argsort(KL_divergence,axis=0)#to see if the result is correct
print(np.argsort(mean_distance,axis=0)[0,:])
print(np.sort(mean_distance,axis=0)[0,:])
print(np.argsort(covariance_distance,axis=0)[0,:])
print(np.sort(covariance_distance,axis=0)[0,:])
print(np.argsort(Hellinger,axis=0)[0,:])
print(np.sort(Hellinger,axis=0)[0,:])


#below is written to automatically detect by replacement of an object which has been assigned two times
eval_kl_argsort=np.argsort(KL_divergence, axis=0)
eval_kl_sort=np.sort(KL_divergence, axis=0)
for iter in range(0,9):
    for k in range(0,KL_divergence.shape[0]):
        if np.where(eval_kl_argsort[0]==k)[0].shape[0]>1:
            if np.sort(KL_divergence,axis=0)[k,np.where(eval_kl_argsort[0]==k)[0][0]]>eval_kl_sort[k,np.where(eval_kl_argsort[0]==k)[0][1]]:
                eval_kl_argsort[0, np.where(eval_kl_argsort[0] == k)[0][0]] = \
                eval_kl_argsort[1, np.where(eval_kl_argsort[0] == k)[0][0]]
            else:
                eval_kl_argsort[0, np.where(eval_kl_argsort[0] == k)[0][1]] = \
                eval_kl_argsort[1, np.where(eval_kl_argsort[0] == k)[0][1]]



print(np.argsort(KL_divergence,axis=0)[0,:])
print(np.sort(KL_divergence,axis=0)[0,:])
print('end')