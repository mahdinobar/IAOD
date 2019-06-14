"""

"""
import open3d
import numpy as np
import copy



def point_cloud(image,depth):
    """

    :param image:
    :param depth:
    :return:
    """
    image_3D = open3d.Image(image)
    depth_3D = open3d.Image(depth)
    rgbd = open3d.create_rgbd_image_from_color_and_depth(image_3D, depth_3D,
                                                               convert_rgb_to_intensity=False)
    fx = 594.21
    fy = 591.04
    a = -0.0030711
    b = 3.3309495
    cx = 339.5
    cy = 242.7
    intrinsic = open3d.PinholeCameraIntrinsic(depth.shape[1], depth.shape[0], fx, fy, cx, cy)
    pcd = open3d.create_point_cloud_from_rgbd_image(rgbd, intrinsic)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # open3d.draw_geometries([pcd])
    return pcd


def point_cloud_with_registration(image_source,depth_source,image_target,depth_target,ratio,
                                  voxel_size = 0.01,threshold = 0.01):
    '''
    This function takes two (image,depth), i.e two rgbd, and computes the point cloud of each corresponding rgbd (also
    by considering intrinsic camera calibration) and then uses global registration to obtain initialization for ICP
    local registration and computes the transformation of the source point cloud (i.e point cloud of the first rgbd)
    by ICP registration result (which is a transfer matrix) and name it pc_3. Then, the datapoints of the resultant
    point cloud (i.e pcd_3) which are far from (i.e distance between points in (x,y,z) ) the source point cloud is
    eliminated from pcd_3 using 'ratio' parameter. Then pcd_3 is returned as output.
    :param image_source:
    :param depth_source:
    :param image_target:
    :param depth_target:
    :param ratio: ratio of maximum distance between datapoints of transformed source point cloud by ICP registration
    and datapoints of target point cloud. [e.g. ratio=0.75 means we only keep datapoints which have distance less
    than 0.75 times the maximum distance]
    :param voxel_size: voxel_size=0.005 means 5cm for dataset
    :param threshold:
    :return:
    '''
    image_3D_source = open3d.Image(image_source)
    depth_3D_source = open3d.Image(depth_source)
    rgbd_source = open3d.create_rgbd_image_from_color_and_depth(image_3D_source, depth_3D_source,
                                                               convert_rgb_to_intensity=False)
    fx = 594.21
    fy = 591.04
    a = -0.0030711
    b = 3.3309495
    cx = 339.5
    cy = 242.7
    intrinsic_source = open3d.PinholeCameraIntrinsic(depth_source.shape[1], depth_source.shape[0], fx, fy, cx, cy)
    pcd_source = open3d.create_point_cloud_from_rgbd_image(rgbd_source, intrinsic_source)
    pcd_source.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    image_3D_target = open3d.Image(image_target)
    depth_3D_target = open3d.Image(depth_target)
    rgbd_target = open3d.create_rgbd_image_from_color_and_depth(image_3D_target, depth_3D_target,
                                                               convert_rgb_to_intensity=False)
    intrinsic_target = open3d.PinholeCameraIntrinsic(depth_target.shape[1], depth_target.shape[0], fx, fy, cx, cy)
    pcd_target = open3d.create_point_cloud_from_rgbd_image(rgbd_target, intrinsic_target)
    pcd_target.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    pcd_source_down, pcd_target_down, pcd_source_fpfh, pcd_target_fpfh = \
            prepare_dataset(pcd_source, pcd_target, voxel_size)

    result_ransac = execute_global_registration(pcd_source_down, pcd_target_down,
            pcd_source_fpfh, pcd_target_fpfh, voxel_size)

    trans_init = result_ransac.transformation
    draw_registration_result(pcd_source, pcd_target, trans_init)
    print("Initial alignment")
    evaluation = open3d.evaluate_registration(pcd_source, pcd_target,
                                       threshold, trans_init)
    print(evaluation)

    print("Apply point-to-point ICP")
    reg_p2p = open3d.registration_icp(pcd_source, pcd_target, threshold, trans_init,
                               open3d.TransformationEstimationPointToPoint())
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    print("")
    draw_registration_result(pcd_source, pcd_target, reg_p2p.transformation)

    pcd_3=copy.deepcopy(pcd_source)
    pcd_3.transform(reg_p2p.transformation.tolist())


    if np.asarray(pcd_3.points).shape[0]>np.asarray(pcd_target.points).shape[0]:
        target_points = np.zeros(np.asarray(pcd_3.points).shape)
        target_points[:np.asarray(pcd_target.points).shape[0], :] = np.asarray(pcd_target.points)
    else:
        target_points = np.asarray(pcd_target.points)[:np.asarray(pcd_3.points).shape[0], :]


    pcd_3_temp = copy.deepcopy(pcd_3)
    # pcd_3.points=open3d.Vector3dVector(np.asarray(pcd_3.points)[
    #     np.linalg.norm(np.asarray(pcd_3.points) - target_points, axis=1) < ratio * np.linalg.norm(
    #         np.asarray(pcd_3.points) - target_points, axis=1).max()])
    pcd_3.points=open3d.Vector3dVector(np.asarray(pcd_3.points)[np.argsort(np.linalg.norm(np.asarray(pcd_3_temp.points) - target_points, axis=1))[
                             0:np.around(ratio * np.asarray(pcd_3_temp.points).shape[0]).astype(int)], :])
    pcd_3.colors = open3d.Vector3dVector(
        np.asarray(pcd_3.colors)[np.argsort(np.linalg.norm(np.asarray(pcd_3_temp.points) - target_points, axis=1))[
                                 0:np.around(ratio * np.asarray(pcd_3_temp.points).shape[0]).astype(int)], :])
    # pcd_3.colors=open3d.Vector3dVector(np.asarray(pcd_3.colors)[
    #     np.linalg.norm(np.asarray(pcd_3_temp.points) - target_points, axis=1) < ratio * np.linalg.norm(
    #         np.asarray(pcd_3_temp.points) - target_points, axis=1).max()])
    open3d.draw_geometries([pcd_3])
    return pcd_3

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    open3d.draw_geometries([source_temp, target_temp])


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = open3d.voxel_down_sample(pcd, voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    open3d.estimate_normals(pcd_down, open3d.KDTreeSearchParamHybrid(
            radius = radius_normal, max_nn = 30))
    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = open3d.compute_fpfh_feature(pcd_down,
            open3d.KDTreeSearchParamHybrid(radius = radius_feature, max_nn = 100))
    return pcd_down, pcd_fpfh

def prepare_dataset(source, target, voxel_size):
    draw_registration_result(source, target, np.identity(4))
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source_down, target_down, source_fpfh, target_fpfh

def execute_global_registration(
        source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = open3d.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            distance_threshold,
            open3d.TransformationEstimationPointToPoint(False), 4,
            [open3d.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            open3d.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
            open3d.RANSACConvergenceCriteria(4000000, 500))
    return result

def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = open3d.registration_icp(source, target, distance_threshold,
            open3d.result_ransac.transformation,
            open3d.TransformationEstimationPointToPlane())
    return result



