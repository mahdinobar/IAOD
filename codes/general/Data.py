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

class data():
    ref_point = []
    cropping = False
    def get_cropped_window(self=None):
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

        image = get_video()
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

        return ref_point


    def get_cropped_image(ref_point):
        array, _ = freenect.sync_get_video()
        array=array[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
        array = cv.cvtColor(array, cv.COLOR_RGB2BGR)
        array = array.astype(np.uint8)
        return array
    def get_cropped_depth(ref_point):
        array, _ = freenect.sync_get_depth(format=freenect.DEPTH_REGISTERED)
        array = array[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
        # print (array.shape)
        array = array.astype(np.uint8)
        return array


    def get_data(cropped_window=None):
        if cropped_window is None:
            ref_point = data.get_cropped_window()
            image = data.get_cropped_image(ref_point)
            depth = data.get_cropped_depth(ref_point)
        else:
            ref_point=cropped_window
            image = data.get_cropped_image(cropped_window)
            depth = data.get_cropped_depth(cropped_window)
        return image,depth,ref_point
