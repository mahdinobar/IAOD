import cv2
import numpy as np
# import open3d
from open3d import *
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.preprocessing import StandardScaler
from sklearn import mixture
import matplotlib.colors
from sklearn.decomposition import PCA
import matplotlib.colors
from sklearn.cluster import DBSCAN, Birch, SpectralClustering, AgglomerativeClustering



def labels_to_color(labels):
    labels_color = np.zeros([labels.shape[0],3])
    color_list = [ 'b', 'g', 'r', 'y','c', 'm', 'k','grey','midnightblue']

    N = 23
    cmap = plt.cm.jet
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    for i in range(0,np.max(labels)+1):
        rgb_color_norm = matplotlib.colors.to_rgb(color_list[i])
        hsv = matplotlib.colors.rgb_to_hsv(rgb_color_norm)
        labels_color[np.where(labels==i),:] = hsv



    return labels_color

def remove_background(p_cloud):
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
    pcd = read_point_cloud('pcd_gmm_raw_data_more_space.ply')
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)



    #
    data = np.hstack((points, colors))

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    #
    foreground = remove_background(data_scaled)
    print(data.shape)
    print(foreground.shape)




    gmm = AgglomerativeClustering(7)


     #Keep only x,y of the point cloud
    labels = gmm.fit_predict(foreground[:,:2])


    # bic.append(gmm.aic(data_scaled))
    labels_color = labels_to_color(labels)
    pcd.colors = Vector3dVector(labels_color)
    pcd.points = Vector3dVector(foreground[:,:3])





    draw_geometries([pcd])