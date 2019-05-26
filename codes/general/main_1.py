"""

"""

import cv2 as cv
import numpy as np
from models import model
from Data import data
from measure_distance import models_distance

# image_source,depth_source, cropped_window=data.get_data(None)
# np.save('image_sequence_1_1_one.npy', image_source)
# np.save('depth_sequence_1_1_one.npy', depth_source)
image_source=np.load('image_sequence_2_1_one.npy')
depth_source=np.load('depth_sequence_2_1_one.npy')
while 1:
    cv.imshow('image_source', image_source)
    key = cv.waitKey(1)
    if key==27:
        break
# image_2_source,depth_2_source,_=data.get_data(cropped_window)
# np.save('image_sequence_1_1_two.npy', image_2_source)
# np.save('depth_sequence_1_1_two.npy', depth_2_source)
image_2_source=np.load('image_sequence_2_2_one.npy')
depth_2_source=np.load('depth_sequence_2_2_one.npy')
while 1:
    cv.imshow('image_2_source', image_2_source)
    key = cv.waitKey(1)
    if key==27:
        break

# image_2_source,depth_2_source,_=data.get_data(cropped_window)
# np.save('image_sequence_1_2_one.npy', image_2_source)
# np.save('depth_sequence_1_2_one.npy', depth_2_source)
image_target=np.load('image_sequence_2_1_two.npy')
depth_target=np.load('depth_sequence_2_1_two.npy')
while 1:
    cv.imshow('image_target', image_target)
    key = cv.waitKey(1)
    if key==27:
        break
# image_2_source,depth_2_source,_=data.get_data(cropped_window)
# np.save('image_sequence_1_2_two.npy', image_2_source)
# np.save('depth_sequence_1_2_two.npy', depth_2_source)
image_2_target=np.load('image_sequence_2_2_two.npy')
depth_2_target=np.load('depth_sequence_2_2_two.npy')
while 1:
    cv.imshow('image_2_target', image_2_target)
    key = cv.waitKey(1)
    if key==27:
        break
#
# image_target,depth_target,_=data.get_data(cropped_window)
# np.save('image_sequence_1_3_one.npy', image_target)
# np.save('depth_sequence_1_3_one.npy', depth_target)
# # image_target=np.load('image_7parts_3.npy')
# # depth_target=np.load('depth_7parts_3.npy')
# while 1:
#     cv.imshow('image_1_3_one', image_target)
#     key = cv.waitKey(1)
#     if key==27:
#         break
# image_2_source,depth_2_source,_=data.get_data(cropped_window)
# np.save('image_sequence_1_3_two.npy', image_2_source)
# np.save('depth_sequence_1_3_two.npy', depth_2_source)
# # image_2_source=np.load('image_6parts.npy')
# # depth_2_source=np.load('depth_6parts.npy')
# while 1:
#     cv.imshow('image_1_3_two', image_2_source)
#     key = cv.waitKey(1)
#     if key==27:
#         break
#
# image_2_target,depth_2_target,_=data.get_data(cropped_window)
# np.save('image_sequence_1_4_one.npy', image_2_target)
# np.save('depth_sequence_1_4_one.npy', depth_2_target)
# # image_2_target=np.load('image_7parts_4.npy')
# # depth_2_target=np.load('depth_7parts_4.npy')
# while 1:
#     cv.imshow('image_1_4_one', image_2_target)
#     key = cv.waitKey(1)
#     if key==27:
#         break
# image_2_source,depth_2_source,_=data.get_data(cropped_window)
# np.save('image_sequence_1_4_two.npy', image_2_source)
# np.save('depth_sequence_1_4_two.npy', depth_2_source)
# # image_2_source=np.load('image_6parts.npy')
# # depth_2_source=np.load('depth_6parts.npy')
# while 1:
#     cv.imshow('image_1_4_two', image_2_source)
#     key = cv.waitKey(1)
#     if key==27:
#         break
#
# image_2_target,depth_2_target,_=data.get_data(cropped_window)
# np.save('image_sequence_1_5_one.npy', image_2_target)
# np.save('depth_sequence_1_5_one.npy', depth_2_target)
# # image_2_target=np.load('image_7parts_4.npy')
# # depth_2_target=np.load('depth_7parts_4.npy')
# while 1:
#     cv.imshow('image_1_5_one', image_2_target)
#     key = cv.waitKey(1)
#     if key==27:
#         break
# image_2_source,depth_2_source,_=data.get_data(cropped_window)
# np.save('image_sequence_1_5_two.npy', image_2_source)
# np.save('depth_sequence_1_5_two.npy', depth_2_source)
# # image_2_source=np.load('image_6parts.npy')
# # depth_2_source=np.load('depth_6parts.npy')
# while 1:
#     cv.imshow('image_1_5_two', image_2_source)
#     key = cv.waitKey(1)
#     if key==27:
#         break
#
# image_2_target,depth_2_target,_=data.get_data(cropped_window)
# np.save('image_sequence_1_6_one.npy', image_2_target)
# np.save('depth_sequence_1_6_one.npy', depth_2_target)
# # image_2_target=np.load('image_7parts_4.npy')
# # depth_2_target=np.load('depth_7parts_4.npy')
# while 1:
#     cv.imshow('image_1_6_one', image_2_target)
#     key = cv.waitKey(1)
#     if key==27:
#         break
# image_2_source,depth_2_source,_=data.get_data(cropped_window)
# np.save('image_sequence_1_6_two.npy', image_2_source)
# np.save('depth_sequence_1_6_two.npy', depth_2_source)
# # image_2_source=np.load('image_6parts.npy')
# # depth_2_source=np.load('depth_6parts.npy')
# while 1:
#     cv.imshow('image_1_6_two', image_2_source)
#     key = cv.waitKey(1)
#     if key==27:
#         break
#
# image_2_target,depth_2_target,_=data.get_data(cropped_window)
# np.save('image_sequence_1_7_one.npy', image_2_target)
# np.save('depth_sequence_1_7_one.npy', depth_2_target)
# # image_2_target=np.load('image_7parts_4.npy')
# # depth_2_target=np.load('depth_7parts_4.npy')
# while 1:
#     cv.imshow('image_1_7_one', image_2_target)
#     key = cv.waitKey(1)
#     if key==27:
#         break
# image_2_source,depth_2_source,_=data.get_data(cropped_window)
# np.save('image_sequence_1_7_two.npy', image_2_source)
# np.save('depth_sequence_1_7_two.npy', depth_2_source)
# # image_2_source=np.load('image_6parts.npy')
# # depth_2_source=np.load('depth_6parts.npy')
# while 1:
#     cv.imshow('image_1_7_two', image_2_source)
#     key = cv.waitKey(1)
#     if key==27:
#         break


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