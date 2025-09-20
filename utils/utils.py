"""
Adapted from https://github.com/thusiyuan/cooperative_scene_parsing/blob/master/utils/sunrgbd_utils.py
"""
import sys

sys.path.append('.')
sys.path.append('..')
import random

import numpy as np
import cv2
import fcl
import numpy as np
import torch as th
import trimesh
from typing import List, Dict


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


def threeD_plane_intersect(planea, planeb):
    """ calculate intersection of two planes in 3D
    a, b   4-tuples/lists
           Ax + By +Cz + D = 0
           A,B,C,D in order  

    output:  line of intersection, np.arrays, shape (3,)
    """
    a_normal, b_normal = np.array(planea[:3]), np.array(planeb[:3])
    a_normal = a_normal / np.linalg.norm(a_normal)
    b_normal = b_normal / np.linalg.norm(b_normal)
    aXb_normal = np.cross(a_normal, b_normal)
    aXb_normal = aXb_normal / np.linalg.norm(aXb_normal)

    A = np.array([a_normal, b_normal, aXb_normal])
    d = np.array([-planea[3], -planeb[3], 0.]).reshape(3, 1)
    # could add np.linalg.det(A) == 0 test to prevent linalg.solve throwing error
    if np.linalg.det(A) == 0:
        print('Two planes are parallel!!!')
        return None, None

    # intersection point:3x1
    p_inter = np.linalg.solve(A, d).T
    return np.concatenate([aXb_normal, p_inter[0]], axis=0)


def reconstrcut_floor_ceiling_from_quad_walls(quad_walls_lst: List[Dict]):
    """
    Reconstruct floor and ceiling from quad walls
    """
    assert len(quad_walls_lst) > 0, 'No quad walls detected!!!'

    all_corners = np.concatenate([np.array(wall_dict['corners']) for wall_dict in quad_walls_lst], axis=0)
    floor_z = np.min(all_corners[:, 2])
    floor_normal = np.array([0, 0, 1])

    ceiling_z = np.max(all_corners[:, 2])
    ceiling_normal = np.array([0, 0, -1])

    gt_floor_points_lst = all_corners[all_corners[:, 2] == floor_z]
    gt_ceil_points_lst = all_corners[all_corners[:, 2] == ceiling_z]

    # build wall meshes
    wall_mesh_lst = []
    for i, wall_dict in enumerate(quad_walls_lst):
        normal = np.array(wall_dict['normal'])
        corners = np.array(wall_dict['corners'])
        wall_mesh = create_spatial_quad_polygen(corners, normal, camera_center=None)
        wall_mesh_lst.append(wall_mesh)

    # room_wall_mesh = trimesh.util.concatenate(wall_mesh_lst)
    # room_wall_mesh.export('room_wall_mesh.ply')

    for i, wall_dict in enumerate(quad_walls_lst):
        wall_floor_points_lst = []
        wall_ceiling_points_lst = []

        i_center = np.array(wall_dict['center'])
        i_normal = np.array(wall_dict['normal'])

        # AX + BY + CZ + D = 0
        D = -np.dot(i_normal, i_center)
        plane_i = np.concatenate([i_normal, [D]], axis=0)

        # find the closest wall to current wall
        # dists = np.abs(np.dot(centers - i_center, i_normal) + D)
        # dists[i] = np.inf
        dists = [check_mesh_distance(wall_mesh_lst[i], wall_mesh_lst[j]) for j in range(len(wall_mesh_lst))]
        dists[i] = np.inf
        # print(f'to Wall {i} dists: {dists}')
        closest_wall_idx = np.argmin(dists)
        dists[closest_wall_idx] = np.inf
        # find the second closest wall to current wall
        second_closest_wall_idx = np.argmin(dists)

        # calculate the intersection of the adjacent walls with the floor and ceiling
        floor_corners = []
        ceiling_corners = []
        for idx in [closest_wall_idx, second_closest_wall_idx]:
            j_normal = np.array(quad_walls_lst[idx]['normal'])
            j_center = np.array(quad_walls_lst[idx]['center'])
            D_j = -np.dot(j_normal, j_center)
            plane_j = np.concatenate([j_normal, [D_j]], axis=0)
            # if two wall are parrallel, skip
            if np.allclose(np.abs(i_normal), np.abs(j_normal), atol=1e-3):
                print(f'Wall {i} and Wall {idx} are parallel!!!')
                print(f'i_normal: {i_normal}, j_normal: {j_normal}')
                continue

            # print(f'Wall {i} and Wall {idx} are adjacent:')
            intersect_line = threeD_plane_intersect(plane_i, plane_j)
            # print('intersect_line', intersect_line)
            intersect_normal = intersect_line[:3]
            intersect_point = intersect_line[3:]

            # floor
            assert np.allclose(abs(np.dot(intersect_normal, floor_normal)), 1,
                               atol=1e-2), 'intersect_normal should be vertical to floor_normal!!!'
            point_on_floor = intersect_point + np.array([0, 0, floor_z - intersect_point[2]])
            # print(f'point_on_floor: {point_on_floor}')
            floor_corners.append(point_on_floor)

            # ceiling
            assert np.allclose(abs(np.dot(intersect_normal, ceiling_normal)), 1,
                               atol=1e-2), 'intersect_normal should be vertical to ceiling_normal!!!'
            point_on_ceiling = intersect_point + np.array([0, 0, ceiling_z - intersect_point[2]])
            # print(f'point_on_ceiling: {point_on_ceiling}')
            ceiling_corners.append(point_on_ceiling)

        if len(floor_corners) == len(ceiling_corners) == 2:
            wall_floor_points_lst.append(np.array(floor_corners))
            wall_ceiling_points_lst.append(np.array(ceiling_corners))

            new_width = np.linalg.norm(floor_corners[0] - floor_corners[1])
            new_height = np.linalg.norm(floor_corners[0] - ceiling_corners[0])
            wall_dict['center'] = ((floor_corners[0] + floor_corners[1] + ceiling_corners[0] + ceiling_corners[1]) /
                                   4).tolist()
            wall_dict['size'] = [float(new_width), 0.01, float(new_height)]
        else:
            print('WARNING: floor/ceiling_corners should have two points!!!')

        # print(f'wall_floor_points_lst: {wall_floor_points_lst}')
        # print(f'wall_ceiling_points_lst: {wall_ceiling_points_lst}')

    # floor_points_lst = np.concatenate(floor_points_lst).reshape(-1, 3)
    # ceiling_points_lst = np.concatenate(ceiling_points_lst).reshape(-1, 3)
    # print(f'floor_points_lst: {floor_points_lst}')
    # print(f'gt_floor_points_lst: {gt_floor_points_lst}')
    # print(f'ceiling_points_lst: {ceiling_points_lst}')
    # print(f'gt_ceil_points_lst: {gt_ceil_points_lst}')
    # return floor_points_lst, ceiling_points_lst


def recover_floor_ceiling_points_from_quad_walls(quad_walls_lst: List[Dict]):
    """
    Reconstruct floor and ceiling points from quad walls
    """
    assert len(quad_walls_lst) > 0, 'No quad walls detected!!!'

    def sort_points(points: np.array):
        """
        Sort points by connecting order
        """
        assert len(points) > 0, 'No points detected!!!'

        start_idx = [0, 1]
        sorted_points = [points[start_idx[0]], points[start_idx[1]]]
        points = np.delete(points, start_idx, axis=0)
        # print(f'deleted points.shape: {points.shape}')
        while len(points) > 0:
            dists = np.linalg.norm(points - sorted_points[-1], axis=1)
            next_idx = np.argmin(dists)
            if next_idx % 2 == 0:
                sorted_points.append(points[next_idx])
                sorted_points.append(points[next_idx + 1])
                del_idx = [next_idx, next_idx + 1]
            else:
                sorted_points.append(points[next_idx - 1])
                sorted_points.append(points[next_idx])
                del_idx = [next_idx - 1, next_idx]
            points = np.delete(points, del_idx, axis=0)
        return np.array(sorted_points)

    all_corners = np.concatenate([wall_dict['corners'] for wall_dict in quad_walls_lst], axis=0)
    floor_z = np.min(all_corners[:, 2])
    raw_floor_points = all_corners[all_corners[:, 2] == floor_z]
    assert (raw_floor_points.shape[0] % 2 == 0)
    print(f'raw_floor_points: {raw_floor_points}')
    # sort points by connecting order
    floor_points = raw_floor_points
    floor_points = sort_points(raw_floor_points)
    print(f'sorted floor_points: {floor_points}')

    ceiling_z = np.max(all_corners[:, 2])
    raw_ceiling_points = all_corners[all_corners[:, 2] == ceiling_z]
    assert (raw_ceiling_points.shape[0] % 2 == 0)
    # print(f'raw_ceiling_points: {raw_ceiling_points}')
    # sort points by connecting order
    ceiling_points = raw_ceiling_points
    ceiling_points = sort_points(raw_ceiling_points)

    return floor_points, ceiling_points


def check_mesh_attachment(object_mesh: trimesh.Trimesh, room_mesh: trimesh.Trimesh):
    """check if the object mesh is attached with the room, i.e. the window/door mesh is collided with the room

    use FCL to detect attachment and collision, see https://github.com/BerkeleyAutomation/python-fcl/ for ref.
    Args:
        object_mesh (o3d.geometry.TriangleMesh): object mesh
        room_walls (o3d.geometry.TriangleMesh): room walls mesh

    Returns:
        bool: True if the object mesh is in the room
    """
    room = fcl.BVHModel()
    room.beginModel(len(room_mesh.vertices), len(room_mesh.faces))
    room.addSubModel(room_mesh.vertices, room_mesh.faces)
    room.endModel()

    window = fcl.BVHModel()
    window.beginModel(len(object_mesh.vertices), len(object_mesh.faces))
    window.addSubModel(object_mesh.vertices, object_mesh.faces)
    window.endModel()

    t1 = fcl.Transform(np.array([1, 0, 0, 0]), np.array([0, 0, 0]))
    t2 = fcl.Transform(np.array([1, 0, 0, 0]), np.array([0., 0, 0]))

    o1 = fcl.CollisionObject(room, t1)
    o2 = fcl.CollisionObject(window, t2)

    request = fcl.CollisionRequest(num_max_contacts=100, enable_contact=True)
    result = fcl.CollisionResult()
    ret = fcl.collide(o1, o2, request, result)
    return result.is_collision


def check_mesh_distance(object_mesh: trimesh.Trimesh, room_mesh: trimesh.Trimesh):
    """caculate distance between object mesh and room mesh, i.e. the window/door mesh is adjacent to the room but is not collided with the room.

    use FCL to detect attachment and collision, see https://github.com/BerkeleyAutomation/python-fcl/ for ref.
    Args:
        object_mesh (o3d.geometry.TriangleMesh): object mesh
        room_walls (o3d.geometry.TriangleMesh): room walls mesh

    Returns:
        float: distance
    """
    room = fcl.BVHModel()
    room.beginModel(len(room_mesh.vertices), len(room_mesh.faces))
    room.addSubModel(room_mesh.vertices, room_mesh.faces)
    room.endModel()

    window = fcl.BVHModel()
    window.beginModel(len(object_mesh.vertices), len(object_mesh.faces))
    window.addSubModel(object_mesh.vertices, object_mesh.faces)
    window.endModel()

    t1 = fcl.Transform(np.array([1, 0, 0, 0]), np.array([0, 0, 0]))
    t2 = fcl.Transform(np.array([1, 0, 0, 0]), np.array([0., 0, 0]))

    o1 = fcl.CollisionObject(room, t1)
    o2 = fcl.CollisionObject(window, t2)

    request = fcl.DistanceRequest()
    result = fcl.DistanceResult()
    ret = fcl.distance(o1, o2, request, result)
    return ret


def my_compute_box_3d(center, size, heading_angle):
    R = np.array([[np.cos(-heading_angle), -np.sin(-heading_angle), 0],
                  [np.sin(-heading_angle), np.cos(-heading_angle), 0], [0, 0, 1]])
    l, w, h = size
    x_corners = [-l, l, l, -l, -l, l, l, -l]
    y_corners = [w, w, -w, -w, w, w, -w, -w]
    z_corners = [h, h, h, h, -h, -h, -h, -h]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] += center[0]
    corners_3d[1, :] += center[1]
    corners_3d[2, :] += center[2]
    return np.transpose(corners_3d)

def heading2rotmat(heading_angle_rad: float) -> np.array:
    """
    Convert z_angle to rotation matrix
    """
    rotmat = np.eye(3)
    cosval = np.cos(heading_angle_rad)
    sinval = np.sin(heading_angle_rad)
    rotmat[0:2, 0:2] = np.array([[cosval, -sinval], [sinval, cosval]])
    return rotmat


def convert_oriented_box_to_trimesh_fmt(box: Dict, color_to_labels: Dict = None) -> trimesh.Trimesh:
    """
    Convert oriented box dict to mesh
    """
    box_center = box['center']
    box_lengths = box['size']
    transform_matrix = np.eye(4)
    transform_matrix[0:3, 3] = box_center
    # only use z angle, rad
    transform_matrix[0:3, 0:3] = heading2rotmat(box['angles'][-1])
    box_trimesh_fmt = trimesh.creation.box(box_lengths, transform_matrix)
    if color_to_labels is not None:
        labels_lst = list(color_to_labels.values())
        colors_lst = list(color_to_labels.keys())
        color = colors_lst[labels_lst.index(box['class'])]
    else:
        color = (np.random.random(3) * 255).astype(np.uint8).tolist()
        # pass
    box_trimesh_fmt.visual.face_colors = color
    return box_trimesh_fmt


def vis_scene_mesh(room_layout_mesh: trimesh.Trimesh,
                   obj_bbox_lst: List[Dict],
                   color_to_labels: Dict = None,
                   room_layout_bbox:trimesh.Trimesh = None) -> trimesh.Trimesh:
    """ visualize scene bbox as mesh

    Args:
        room_layout_mesh (trimesh.Trimesh): closed mesh of room layout
        obj_bbox_lst (List[Dict]): object bounding box list
        color_to_labels (Dict, optional): color for object categories. Defaults to None.
        room_layout_bbox (trimesh.Trimesh): _description_. Defaults to None.

    Returns:
        trimesh.Trimesh: _description_
    """

    def create_oriented_bbox(scene_bbox: List[Dict]) -> trimesh.Trimesh:
        """Export oriented (around Z axis) scene bbox to meshes
        Args:
            scene_bbox: (N x 7 numpy array): xyz pos of center and 3 lengths (dx,dy,dz)
                and heading angle around Z axis.
                Y forward, X right, Z upward. heading angle of positive X is 0,
                heading angle of positive Y is 90 degrees.
            out_filename: (string) filename
        """
        scene = trimesh.scene.Scene()
        for box in scene_bbox:
            scene.add_geometry(convert_oriented_box_to_trimesh_fmt(box, color_to_labels))

        mesh_list = trimesh.util.concatenate(scene.dump())
        return mesh_list

    v_object_meshes = create_oriented_bbox(obj_bbox_lst)
    if room_layout_bbox is not None:
        scene_mesh = trimesh.util.concatenate([room_layout_mesh, v_object_meshes, room_layout_bbox])
    elif room_layout_mesh is not None:
        scene_mesh = trimesh.util.concatenate([room_layout_mesh, v_object_meshes])
    else:
        scene_mesh = trimesh.util.concatenate([v_object_meshes])
    return scene_mesh

def create_spatial_quad_polygen(quad_vertices: List, normal: np.array, camera_center: np.array):
    """create a quad polygen for spatial mesh
    """
    if camera_center is None:
        camera_center = np.array([0, 0, 0])
    quad_vertices = (quad_vertices - camera_center)
    quad_triangles = []
    triangle = np.array([[0, 2, 1], [2, 0, 3]])
    quad_triangles.append(triangle)

    quad_triangles = np.concatenate(quad_triangles, axis=0)

    mesh = trimesh.Trimesh(vertices=quad_vertices,
                           faces=quad_triangles,
                           vertex_normals=np.tile(normal, (4, 1)),
                           process=False)

    centroid = np.mean(quad_vertices, axis=0)
    # print(f'centroid: {centroid}')
    normal_point = centroid + np.array(normal) * 0.5
    # print(f'normal_point: {normal_point}')

    # pcd_o3d = open3d.geometry.PointCloud()
    # pcd_o3d.points = open3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    # pcd_o3d.points.append(normal_point)
    # pcd_o3d.points.append(centroid)
    # return mesh, pcd_o3d
    return mesh

import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=Loader)
    return config

def HWC3(x: np.ndarray):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def resize_image(input_image: np.ndarray, resolution: int):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img


def nms(x, t, s):
    x = cv2.GaussianBlur(x.astype(np.float32), (0, 0), s)

    f1 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
    f2 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
    f3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
    f4 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)

    y = np.zeros_like(x)

    for f in [f1, f2, f3, f4]:
        np.putmask(y, cv2.dilate(x, kernel=f) == x, x)

    z = np.zeros_like(y, dtype=np.uint8)
    z[y > t] = 255
    return z


def make_noise_disk(H, W, C, F):
    noise = np.random.uniform(low=0, high=1, size=((H // F) + 2, (W // F) + 2, C))
    noise = cv2.resize(noise, (W + 2 * F, H + 2 * F), interpolation=cv2.INTER_CUBIC)
    noise = noise[F: F + H, F: F + W]
    noise -= np.min(noise)
    noise /= np.max(noise)
    if C == 1:
        noise = noise[:, :, None]
    return noise


def min_max_norm(x):
    x -= np.min(x)
    x /= np.maximum(np.max(x), 1e-5)
    return x


def safe_step(x, step=2):
    y = x.astype(np.float32) * float(step + 1)
    y = y.astype(np.int32).astype(np.float32) / float(step)
    return y


def img2mask(img, H, W, low=10, high=90):
    assert img.ndim == 3 or img.ndim == 2
    assert img.dtype == np.uint8

    if img.ndim == 3:
        y = img[:, :, random.randrange(0, img.shape[2])]
    else:
        y = img

    y = cv2.resize(y, (W, H), interpolation=cv2.INTER_CUBIC)

    if random.uniform(0, 1) < 0.5:
        y = 255 - y

    return y < np.percentile(y, random.randrange(low, high))
