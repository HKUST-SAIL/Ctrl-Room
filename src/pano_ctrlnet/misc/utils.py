"""
Adapted from https://github.com/thusiyuan/cooperative_scene_parsing/blob/master/utils/sunrgbd_utils.py
"""
import numpy as np
import torch as th


def normalize(vector):
    return vector / np.linalg.norm(vector)


def parse_camera_info(camera_info, height, width):
    """ extract intrinsic and extrinsic matrix
    """
    lookat = normalize(camera_info[3:6])
    up = normalize(camera_info[6:9])

    W = lookat
    U = np.cross(W, up)
    V = -np.cross(W, U)

    rot = np.vstack((U, V, W))
    trans = camera_info[:3]

    xfov = camera_info[9]
    yfov = camera_info[10]

    K = np.diag([1, 1, 1])

    K[0, 2] = width / 2
    K[1, 2] = height / 2

    K[0, 0] = K[0, 2] / np.tan(xfov)
    K[1, 1] = K[1, 2] / np.tan(yfov)

    return rot, trans, K


def flip_towards_viewer(normals, points):
    points = points / np.linalg.norm(points)
    proj = points.dot(normals[:2, :].T)
    flip = np.where(proj > 0)
    normals[flip, :] = -normals[flip, :]
    return normals


def get_corners_of_bb3d(basis, coeffs, centroid):
    corners = np.zeros((8, 3))
    # order the basis
    index = np.argsort(np.abs(basis[:, 0]))[::-1]
    # the case that two same value appear the same time
    if index[2] != 2:
        index[1:] = index[1:][::-1]
    basis = basis[index, :]
    coeffs = coeffs[index]
    # Now, we know the basis vectors are orders X, Y, Z. Next, flip the basis vectors towards the viewer
    basis = flip_towards_viewer(basis, centroid)
    coeffs = np.abs(coeffs)
    corners[0, :] = -basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[1, :] = basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[2, :] = basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[3, :] = -basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]

    corners[4, :] = -basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]
    corners[5, :] = basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]
    corners[6, :] = basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]
    corners[7, :] = -basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]
    corners = corners + np.tile(centroid, (8, 1))
    return corners


def get_corners_of_bb3d_no_index(basis, coeffs, centroid):
    corners = np.zeros((8, 3))
    coeffs = np.abs(coeffs)
    corners[0, :] = -basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[1, :] = basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[2, :] = basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[3, :] = -basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]

    corners[4, :] = -basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]
    corners[5, :] = basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]
    corners[6, :] = basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]
    corners[7, :] = -basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]

    corners = corners + np.tile(centroid, (8, 1))
    return corners


def project_3d_points_to_2d(points3d, R_ex, K):
    """
    Project 3d points from camera-centered coordinate to 2D image plane
    Parameters
    ----------
    points3d: numpy array
        3d location of point
    R_ex: numpy array
        extrinsic camera parameter
    K: numpy array
        intrinsic camera parameter
    Returns
    -------
    points2d: numpy array
        2d location of the point
    """
    points3d = R_ex.dot(points3d.T).T
    x3 = points3d[:, 0]
    y3 = -points3d[:, 1]
    z3 = np.abs(points3d[:, 2])
    xx = x3 * K[0, 0] / z3 + K[0, 2]
    yy = y3 * K[1, 1] / z3 + K[1, 2]
    points2d = np.vstack((xx, yy))
    return points2d


def project_struct_bdb_to_2d(basis, coeffs, center, R_ex, K):
    """
    Project 3d bounding box to 2d bounding box
    Parameters
    ----------
    basis, coeffs, center, R_ex, K
        : K is the intrinsic camera parameter matrix
        : Rtilt is the extrinsic camera parameter matrix in right hand coordinates
    Returns
    -------
    bdb2d: dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    """
    corners3d = get_corners_of_bb3d(basis, coeffs, center)
    corners = project_3d_points_to_2d(corners3d, R_ex, K)
    bdb2d = dict()
    bdb2d['x1'] = int(max(np.min(corners[0, :]), 1))  # x1
    bdb2d['y1'] = int(max(np.min(corners[1, :]), 1))  # y1
    bdb2d['x2'] = int(min(np.max(corners[0, :]), 2 * K[0, 2]))  # x2
    bdb2d['y2'] = int(min(np.max(corners[1, :]), 2 * K[1, 2]))  # y2
    # if not check_bdb(bdb2d, 2*K[0, 2], 2*K[1, 2]):
    #     bdb2d = None
    return bdb2d


def matrix_to_euler_angles(R_w_i: np.array):
    """
    Convert rotation matrix to euler angles in Z-X-Y order
    """
    roll = np.arcsin(R_w_i[2, 1])
    pitch = np.arctan2(-R_w_i[2, 0] / np.cos(roll), R_w_i[2, 2] / np.cos(roll))
    yaw = np.arctan2(-R_w_i[0, 1] / np.cos(roll), R_w_i[1, 1] / np.cos(roll))

    return np.array([roll, pitch, yaw])


def euler_angle_to_matrix(angles_lst: (list, th.Tensor)):
    """ rotation matrix in Z(yaw)-X(roll)-Y(pitch) order R_w_i

    Args:
        angles_lst (_type_): _description_

    Returns:
        _type_: _description_
    """

    if isinstance(angles_lst, list):
        roll = angles_lst[0]
        pitch = angles_lst[1]
        yaw = angles_lst[2]
        R = np.array([[
            np.cos(yaw) * np.cos(pitch) - np.sin(roll) * np.sin(yaw) * np.sin(pitch), -np.cos(roll) * np.sin(yaw),
            np.cos(yaw) * np.sin(pitch) + np.cos(pitch) * np.sin(roll) * np.sin(yaw)
        ],
                      [
                          np.cos(pitch) * np.sin(yaw) + np.cos(yaw) * np.sin(roll) * np.sin(pitch),
                          np.cos(roll) * np.cos(yaw),
                          np.sin(yaw) * np.sin(pitch) - np.cos(yaw) * np.sin(roll) * np.cos(pitch)
                      ], [-np.cos(roll) * np.sin(pitch),
                          np.sin(roll), np.cos(roll) * np.cos(pitch)]])
    elif isinstance(angles_lst, th.Tensor):
        if len(angles_lst.shape) == 1:
            roll = angles_lst[0]
            pitch = angles_lst[1]
            yaw = angles_lst[2]
            R = th.zeros((3, 3))
            R[0, 0] = th.cos(yaw) * th.cos(pitch) - th.sin(roll) * th.sin(yaw) * th.sin(pitch)
            R[0, 1] = -th.cos(roll) * th.sin(yaw)
            R[0, 2] = th.cos(yaw) * th.sin(pitch) + th.cos(pitch) * th.sin(roll) * th.sin(yaw)
            R[1, 0] = th.cos(pitch) * th.sin(yaw) + th.cos(yaw) * th.sin(roll) * th.sin(pitch)
            R[1, 1] = th.cos(roll) * th.cos(yaw)
            R[1, 2] = th.sin(yaw) * th.sin(pitch) - th.cos(yaw) * th.sin(roll) * th.cos(pitch)
            R[2, 0] = -th.cos(roll) * th.sin(pitch)
            R[2, 1] = th.sin(roll)
            R[2, 2] = th.cos(roll) * th.cos(pitch)
            R = R.to(angles_lst.device)
        elif len(angles_lst.shape) == 3:
            B, C, _ = angles_lst.shape
            roll = angles_lst[:, :, 0]
            pitch = angles_lst[:, :, 1]
            yaw = angles_lst[:, :, 2]
            R = th.zeros((B, C, 3, 3))
            R[:, :, 0, 0] = th.cos(yaw) * th.cos(pitch) - th.sin(roll) * th.sin(yaw) * th.sin(pitch)
            R[:, :, 0, 1] = -th.cos(roll) * th.sin(yaw)
            R[:, :, 0, 2] = th.cos(yaw) * th.sin(pitch) + th.cos(pitch) * th.sin(roll) * th.sin(yaw)
            R[:, :, 1, 0] = th.cos(pitch) * th.sin(yaw) + th.cos(yaw) * th.sin(roll) * th.sin(pitch)
            R[:, :, 1, 1] = th.cos(roll) * th.cos(yaw)
            R[:, :, 1, 2] = th.sin(yaw) * th.sin(pitch) - th.cos(yaw) * th.sin(roll) * th.cos(pitch)
            R[:, :, 2, 0] = -th.cos(roll) * th.sin(pitch)
            R[:, :, 2, 1] = th.sin(roll)
            R[:, :, 2, 2] = th.cos(roll) * th.cos(pitch)
            R = R.to(angles_lst.device)
        # print(R)
    else:
        raise NotImplementedError
    return R
