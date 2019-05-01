"""

"""
import open3d



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



