import cv2
import numpy as np
from open3d import *
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.preprocessing import StandardScaler
from sklearn import mixture
import matplotlib.colors
from sklearn.decomposition import PCA
import matplotlib.colors
from sklearn.cluster import DBSCAN, Birch, SpectralClustering, AgglomerativeClustering
from sklearn import metrics



def labels_to_color(labels):
    labels_color = np.zeros([labels.shape[0],3])
    color_list = ['m', 'k','grey','midnightblue', 'b', 'g', 'r', 'y','c']

    N = 23
    cmap = plt.cm.jet
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    for i in range(0,np.max(labels)+1):
        rgb_color_norm = matplotlib.colors.to_rgb(color_list[i])
        # hsv = matplotlib.colors.rgb_to_hsv(rgb_color_norm)
        labels_color[np.where(labels==i),:] = rgb_color_norm



    return labels_color

def remove_background(p_cloud):
    #p_cloud is an array here
    gmm = mixture.GaussianMixture(n_components=2,
                                  covariance_type="spherical")

    labels = gmm.fit_predict(p_cloud)
    average_z = np.zeros((2,1))
    for i in range(0,np.max(labels)+1):
        average_z[i] = np.average(p_cloud[np.where(labels == i), 2])
    print(average_z)
    background_label= np.argmin(average_z)
    foreground = np.delete(p_cloud,np.where(labels ==background_label),axis=0)
    return foreground







if __name__ == "__main__":
    pcd = read_point_cloud('pcd_bgm_raw_data_more_space.ply')
    # pcd = read_point_cloud('BGM_5objects_far.ply')
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    draw_geometries([pcd])

    #
    data = np.hstack((points, colors))

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)


    foreground = remove_background(data_scaled)
    print(data.shape)
    print(foreground.shape)

    best_bic = -np.infty
    best_gmm = None

    for i in range(0,5):
        gmm = mixture.BayesianGaussianMixture(5,covariance_type='spherical',init_params='random')

        lab = gmm.fit_predict(foreground[:,:2])
        current_bic = gmm.lower_bound_#metrics.davies_bouldin_score(foreground[:,:2], lab)
        print(i)
        if current_bic>best_bic:
            print(current_bic)
            best_bic = current_bic
            best_gmm = gmm




    labels = best_gmm.fit_predict(foreground[:,:2])


    labels_color = labels_to_color(labels)
    pcd.colors = Vector3dVector(labels_color)
    pcd.points = Vector3dVector(foreground[:,:3])


    draw_geometries([pcd])