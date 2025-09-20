import os
import sys

sys.path.append('.')
sys.path.append('..')

import argparse

import json
import numpy as np
from tqdm import tqdm
import imghdr
import shutil
import cv2
import trimesh
import copy
import multiprocessing as mp
from collections import Counter, OrderedDict
from PIL import Image

import open3d as o3d
from panda3d.core import Triangulator
from typing import List, Tuple, Dict, Any, Union

from utils.utils import (matrix_to_euler_angles, create_spatial_quad_polygen, my_compute_box_3d,
                        check_mesh_attachment, check_mesh_distance)
from utils.equirect_projection import vis_objs3d, vis_floor_ceiling
from dataset.metadata import (INVALID_SCENES_LST, INVALID_ROOMS_LST, OBJECT_LABEL_IDS, ROOM_WALLS_LARGER_THAN_10,
                              COLOR_TO_ADEK_LABEL, ST3D_LIVINGROOM_MIN_LEN, ST3D_LIVINGROOM_QUAD_WALL_MAX_LEN,
                              ST3D_BEDROOM_QUAD_WALL_MAX_LEN)

from dataset.st3d_dataset import get_mesh_from_corners, np_coorx2u, np_coory2v, find_occlusion, corners_to_1d_boundary
from dataset.st3d.st3d_scene import St3dRoom, ST3DDataset
from dataset.gen_scene_text import get_scene_description

'''
Assume datas is extracted by `misc/structured3d_extract_zip.py`.
That is to said, assuming following structure:
- {in_root}/scene_xxxxx
    - rgb/
        - *png
    - layout/
        - *txt


The reorganized structure as follow:
- {out_train_root}
    - img/             # rgb panoramas
        - scene_xxxxx_*png
    - label_cor/       # 2D layout coordinates
        - scene_xxxxx_*txt
    - quad_walls/   # 3D quad walls 
        - scene_xxxxx_*txt
    - cam_pos/          # camera position
        - scene_xxxxx_*txt
    - room_type/        # room type
        - scene_xxxxx_*txt
    - bbox_3d/          # 3D bbox of objects
        - scene_xxxxx_*json 

- {out_valid_root} ...
- {out_test_root} ...
'''
ALL_SCENE = ['scene_%05d' % i for i in range(0, 3500)]
TRAIN_SCENE = ['scene_%05d' % i for i in range(0, 3000)]
TEST_SCENE = ['scene_%05d' % i for i in range(3000, 3500)]

ST3D_BEDROOM_FURNITURES_SET = set()
ST3D_LIVINGROOM_FURNITURES_SET = set()
ST3D_DININGROOM_FURNITURES_SET = set()
ST3D_KITCHEN_FURNITURES_SET = set()
ST3D_STUDY_FURNITURES_SET = set()
ST3D_BATHROOM_FURNITURES_SET = set()


def heading2rotmat(heading_angle_rad):
    rotmat = np.eye(3)
    cosval = np.cos(heading_angle_rad)
    sinval = np.sin(heading_angle_rad)
    rotmat[0:2, 0:2] = np.array([[cosval, -sinval], [sinval, cosval]])

    # # rot around y axis
    # rotmat = np.eye(3)
    # cosval = np.cos(heading_angle_rad)
    # sinval = np.sin(heading_angle_rad)
    # rotmat[0, 0] = cosval
    # rotmat[0, 2] = sinval
    # rotmat[2, 0] = -sinval
    # rotmat[2, 2] = cosval
    return rotmat


def convert_oriented_box_to_trimesh_fmt(box: Dict, color_to_labels: Dict = None):
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
                   room_layout_bbox=None) -> trimesh.Trimesh:

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


def vis_color_pointcloud(rgb_img_filepath, depth_img_filepath, saved_color_pcl_filepath):

    def get_unit_spherical_map():
        h = 512
        w = 1024

        coorx, coory = np.meshgrid(np.arange(w), np.arange(h))
        us = np_coorx2u(coorx, w)
        vs = np_coory2v(coory, h)

        X = np.expand_dims(np.cos(vs) * np.sin(us), 2)
        Y = np.expand_dims(np.sin(vs), 2)
        Z = np.expand_dims(np.cos(vs) * np.cos(us), 2)
        unit_map = np.concatenate([X, Z, Y], axis=2)

        return unit_map

    def display_inlier_outlier(cloud, ind):
        inlier_cloud = cloud.select_by_index(ind)
        outlier_cloud = cloud.select_by_index(ind, invert=True)

        print("Showing outliers (red) and inliers (gray): ")
        outlier_cloud.paint_uniform_color([1, 0, 0])
        inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
        o3d.visualization.draw([inlier_cloud, outlier_cloud])

    assert os.path.exists(rgb_img_filepath), 'rgb panorama doesnt exist!!!'
    assert os.path.exists(depth_img_filepath), 'depth panorama doesnt exist!!!'

    raw_depth_img = cv2.imread(depth_img_filepath, cv2.IMREAD_UNCHANGED)
    if len(raw_depth_img.shape) == 3:
        raw_depth_img = cv2.cvtColor(raw_depth_img, cv2.COLOR_BGR2GRAY)
    depth_img = np.asarray(raw_depth_img)
    if np.isnan(depth_img.any()) or len(depth_img[depth_img > 0]) == 0:
        print('empyt depth image')
        exit(-1)

    raw_rgb_img = cv2.imread(rgb_img_filepath, cv2.IMREAD_UNCHANGED)
    rgb_img = cv2.cvtColor(raw_rgb_img, cv2.COLOR_BGR2RGB)
    if rgb_img.shape[2] == 4:
        rgb_img = rgb_img[:, :, :3]
    if np.isnan(rgb_img.any()) or len(rgb_img[rgb_img > 0]) == 0:
        print('empyt rgb image')
        exit(-1)
    color = np.clip(rgb_img, 0.0, 255.0) / 255.0
    # print(f'raw_rgb shape: {rgb_img.shape} color shape: {color.shape}, ')

    depth_img = np.expand_dims((depth_img / 1000.0), axis=2)
    pointcloud = depth_img * get_unit_spherical_map()

    o3d_pointcloud = o3d.geometry.PointCloud()
    o3d_pointcloud.points = o3d.utility.Vector3dVector(pointcloud.reshape(-1, 3))
    o3d_pointcloud.colors = o3d.utility.Vector3dVector(color.reshape(-1, 3))
    # remove outliers
    # cl, ind = o3d_pointcloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    # display_inlier_outlier(o3d_pointcloud, ind)
    o3d.io.write_point_cloud(saved_color_pcl_filepath, o3d_pointcloud)
    return o3d_pointcloud

def verify_normal(corner_i, corner_j, delta_height, plane_normal):
    edge_a = corner_j + delta_height - corner_i
    edge_b = delta_height

    normal = np.cross(edge_a, edge_b)
    normal /= np.linalg.norm(normal, ord=2)

    inner_product = normal.dot(plane_normal)

    if inner_product > 1e-8:
        return False
    else:
        return True


def convert_lines_to_vertices(lines):
    """convert line representation to polygon vertices
    """
    polygons = []
    lines = np.array(lines)

    polygon = None
    while len(lines) != 0:
        if polygon is None:
            polygon = lines[0].tolist()
            lines = np.delete(lines, 0, 0)

        lineID, juncID = np.where(lines == polygon[-1])
        vertex = lines[lineID[0], 1 - juncID[0]]
        lines = np.delete(lines, lineID, 0)

        if vertex in polygon:
            polygons.append(polygon)
            polygon = None
        else:
            polygon.append(vertex)

    return polygons

def parse_bbox_in_room(room_folderpath: str, room_layout_mesh: trimesh.Trimesh, new_labeld_room_filepath: str = None):
    """ parse object bounding box in room

    Args:
        room_folderpath (str): room folder path
        room_layout_mesh (_type_): room layout mesh derived from 2d layout
        quad_walls_dict (Dict[str, List]): room layout as quad walls.
        new_labeld_room_filepath (str): new labeled room filepath, labelCloud format.
    Returns:
        _type_: _description_
    """

    room_bbox_3d_path = os.path.join(room_folderpath, 'full', 'bbox_3d.json')
    rgb_img_path = os.path.join(room_folderpath, 'full', 'rgb_rawlight.png')
    instance_img_path = os.path.join(room_folderpath, 'full', 'instance.png')
    camera_pos_path = os.path.join(room_folderpath, 'camera_xyz.txt')

    # parse room bbox
    if new_labeld_room_filepath is not None and os.path.exists(new_labeld_room_filepath):
        room_bbox_3d_path = new_labeld_room_filepath
    with open(room_bbox_3d_path, 'r') as file:
        room_anno_3d_dict = json.load(file)

    rgb_img = cv2.imread(rgb_img_path, cv2.IMREAD_UNCHANGED)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    instance_img = cv2.imread(instance_img_path, cv2.IMREAD_UNCHANGED)
    cam_position = np.loadtxt(camera_pos_path)

    layout_bbox_min = trimesh.bounds.corners(room_layout_mesh.bounding_box_oriented.bounds).min(axis=0)
    layout_bbox_max = trimesh.bounds.corners(room_layout_mesh.bounding_box_oriented.bounds).max(axis=0)
    layout_bbox_size = layout_bbox_max - layout_bbox_min
    layout_bbox = room_layout_mesh.bounding_box_oriented

    def check_bbox_in_room(bbox: Dict, room_layout_mesh: trimesh.Trimesh, layout_bbox_min: np.array,
                           layout_bbox_max: np.array):
        bbox_center = np.array([bbox['center']])
        margin_dist = 0.10
        # if object bbox center is outside of room layout
        if bbox_center[:, 0] < layout_bbox_min[0] or bbox_center[:, 0] > layout_bbox_max[0] or \
            bbox_center[:, 1] < layout_bbox_min[1] or bbox_center[:, 1] > layout_bbox_max[1] or \
            bbox_center[:, 2] < layout_bbox_min[2] or bbox_center[:, 2] > layout_bbox_max[2]:
            obj_bbox_mesh = convert_oriented_box_to_trimesh_fmt(bbox)
            if check_mesh_attachment(obj_bbox_mesh,
                                     room_layout_mesh) and bbox['class'] in ['door', 'window', 'picture', 'curtain']:
                # print(f'{bbox["class"]} is attached to room ')
                return True
            else:
                min_distance = check_mesh_distance(obj_bbox_mesh, room_layout_mesh)
                if min_distance < margin_dist and bbox['class'] in ['door', 'window', 'picture', 'curtain']:
                    # print(f'{bbox["class"]} is close to room, distance {min_distance} ')
                    return True
            return False
        else:
            return True

    # object bboxs in camera frame
    obj_bbox_lst = []
    # normalized object bboxs
    obj_bbox_normal_lst = []

    if new_labeld_room_filepath is not None:
        # if new annotated room file exists, we use it
        # the new annotated room file is in labelCloud format
        objects_bbox_lst = room_anno_3d_dict['objects']
        for bbox in objects_bbox_lst:
            angles = np.array(
                [float(bbox['rotations']['x']),
                 float(bbox['rotations']['y']),
                 float(bbox['rotations']['z'])])
            bbox_size = np.array([
                float(bbox['dimensions']['length']),
                float(bbox['dimensions']['width']),
                float(bbox['dimensions']['height'])
            ])
            bbox_center = np.array(
                [float(bbox['centroid']['x']),
                 float(bbox['centroid']['y']),
                 float(bbox['centroid']['z'])])

            obj_bbox_dict = {}
            obj_bbox_dict['class'] = bbox['name']
            obj_bbox_dict['angles'] = angles.tolist()
            obj_bbox_dict['center'] = bbox_center.tolist()
            obj_bbox_dict['size'] = bbox_size.tolist()
            obj_bbox_dict['corners'] = my_compute_box_3d(bbox_center, bbox_size / 2, -angles[2]).tolist()
            if check_bbox_in_room(obj_bbox_dict, room_layout_mesh, layout_bbox_min, layout_bbox_max):
                obj_bbox_lst.append(obj_bbox_dict)

                obj_bbox_normal_dict = {}
                obj_bbox_normal_dict['class'] = bbox['name']
                obj_bbox_normal_dict['angles'] = [np.cos(angles[2]), np.sin(angles[2])]
                # here we normalize bbox center and size w.r.t. room_layout bbox
                # normalize bbox center to [-1, 1]
                obj_bbox_normal_dict['center'] = (bbox_center / layout_bbox_size).tolist()
                # normalize bbox size to [-1, 1]
                obj_bbox_normal_dict['size'] = (bbox_size / layout_bbox_size).tolist()
                obj_bbox_normal_lst.append(obj_bbox_normal_dict)
    else:
        id2index = dict()
        for index, object in enumerate(room_anno_3d_dict):
            id2index[object.get('ID')] = index
        # skip background
        for index in np.unique(instance_img)[:-1]:
            # for each instance in current image
            # we remove some incorrect objeect labels manually
            if index not in id2index.keys():
                continue
            bbox = room_anno_3d_dict[id2index[index]]

            if bbox['label'] not in OBJECT_LABEL_IDS.keys():
                continue

            basis = np.array(bbox['basis'])
            coeffs = np.array(bbox['coeffs'])
            centroid = np.array(bbox['centroid'])

            obj_bbox_dict = {}
            obj_bbox_dict['rotations'] = basis.tolist()
            obj_bbox_dict['centroid'] = list(centroid)
            obj_bbox_dict['dimensions'] = list(coeffs)
            obj_bbox_dict['class'] = bbox['label']

            rotation_euler_angles_rad = matrix_to_euler_angles(basis)
            obj_bbox_dict['angles'] = rotation_euler_angles_rad.tolist()
            bbox_center = (centroid - cam_position) * 0.001
            obj_bbox_dict['center'] = bbox_center.tolist()
            bbox_size = coeffs * 0.001 * 2
            obj_bbox_dict['size'] = bbox_size.tolist()
            obj_bbox_dict['corners'] = my_compute_box_3d(bbox_center, bbox_size / 2,
                                                         -rotation_euler_angles_rad[2]).tolist()
            if check_bbox_in_room(obj_bbox_dict, room_layout_mesh, layout_bbox_min, layout_bbox_max):
                obj_bbox_lst.append(obj_bbox_dict)

                obj_bbox_normal_dict = {}
                obj_bbox_normal_dict['class'] = bbox['label']
                obj_bbox_normal_dict['angles'] = [
                    np.cos(rotation_euler_angles_rad[2]),
                    np.sin(rotation_euler_angles_rad[2])
                ]
                # here we normalize bbox center and size w.r.t. room_layout bbox
                # normalize bbox center to [-1, 1]
                obj_bbox_normal_dict['center'] = (bbox_center / layout_bbox_size).tolist()
                # normalize bbox size to [-1, 1]
                obj_bbox_normal_dict['size'] = (bbox_size / layout_bbox_size).tolist()
                obj_bbox_normal_lst.append(obj_bbox_normal_dict)

    if len(obj_bbox_lst) < ST3D_LIVINGROOM_MIN_LEN:
        return None, None

    return {"objects": obj_bbox_lst}, {"objects": obj_bbox_normal_lst}


def parse_room_layout(img_filepath: str, cam_pos_filepath: str, layout_coor_filepath: str) -> trimesh.Trimesh:
    # Read image
    equirect_img = cv2.imread(img_filepath, cv2.IMREAD_UNCHANGED)
    equirect_img = cv2.cvtColor(equirect_img, cv2.COLOR_BGR2RGB)
    if equirect_img.shape[2] == 4:
        equirect_img = equirect_img[:, :, :3]
    H, W = equirect_img.shape[:2]

    # read camera position file
    cam_pos_lst = []
    with open(cam_pos_filepath) as f:
        cam_pos_lst = np.array([line.strip().split() for line in f if line.strip()], np.float32)
    assert len(cam_pos_lst) == 1, cam_pos_filepath
    # convert the unit into meter
    cam_pos_lst = cam_pos_lst[0] * 0.001

    # Read ground truth corners
    corners_lst = []
    with open(layout_coor_filepath, 'r') as f:
        corners_lst = np.array([line.strip().split() for line in f if line.strip()], np.float32)

        # Corner with minimum x should at the beginning
        corners_lst = np.roll(corners_lst[:, :2], -2 * np.argmin(corners_lst[::2, 0]), 0)

    points, faces = get_mesh_from_corners(corners_lst,
                                          H,
                                          W,
                                          camera_position=cam_pos_lst,
                                          rgb_img=equirect_img,
                                          b_ignore_floor=False,
                                          b_ignore_ceiling=False,
                                          b_ignore_wall=False,
                                          b_in_world_frame=False)
    # print(f'points.shape: {points.shape}, faces.shape: {faces.shape}')
    # downsample the mesh
    raw_mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(points[:, :3]),
                                         triangles=o3d.utility.Vector3iVector(faces))
    simplified_mesh = raw_mesh.simplify_vertex_clustering(voxel_size=0.2,
                                                          contraction=o3d.geometry.SimplificationContraction.Average)
    simplified_vertices = simplified_mesh.vertices
    simplified_faces = np.array(simplified_mesh.triangles)
    room_layout_mesh = trimesh.Trimesh(vertices=simplified_vertices, faces=simplified_faces, process=True)
    return room_layout_mesh


def create_layout_mesh(vertices, vertices_floor, delta_height, camera_center):
    # create mesh for 3D floorplan visualization
    triangles = []

    # the number of vertical walls
    num_walls = len(vertices)

    # 1. vertical wall (always rectangle)
    num_vertices = 0
    for i in range(len(vertices)):
        # hardcode triangles for each vertical wall
        triangle = np.array([[0, 2, 1], [2, 0, 3]])
        triangles.append(triangle + num_vertices)
        num_vertices += 4

    # 2. floor and ceiling
    # Since the floor and ceiling may not be a rectangle, triangulate the polygon first.
    tri = Triangulator()
    for i in range(len(vertices_floor)):
        tri.add_vertex(vertices_floor[i, 0], vertices_floor[i, 1])

    for i in range(len(vertices_floor)):
        tri.add_polygon_vertex(i)

    tri.triangulate()

    # polygon triangulation
    triangle = []
    for i in range(tri.getNumTriangles()):
        triangle.append([tri.get_triangle_v0(i), tri.get_triangle_v1(i), tri.get_triangle_v2(i)])
    triangle = np.array(triangle)

    # add triangles for floor and ceiling
    triangles.append(triangle + num_vertices)
    num_vertices += len(np.unique(triangle))
    triangles.append(triangle + num_vertices)

    # 3. Merge wall, floor, and ceiling
    vertices.append(vertices_floor)
    vertices.append(vertices_floor + delta_height)
    vertices = np.concatenate(vertices, axis=0)

    triangles = np.concatenate(triangles, axis=0)

    # 4. create mesh
    vertices_in_cam = (vertices - camera_center) * 0.001 if camera_center is not None else vertices * 0.001
    mesh = trimesh.Trimesh(vertices=vertices_in_cam, faces=triangles, process=False)
    mesh = mesh.subdivide_to_size(0.02, max_iter=1000)
    return mesh


def merge_walls(quad_wall_dict_lst: List[Dict], quad_wall_mesh_lst: List[trimesh.Trimesh]):

    def cat_mesh(m1, m2):
        v1, f1 = m1
        v2, f2 = m2
        v = np.vstack([v1, v2])
        f = np.vstack([f1, f2 + len(v1)])
        return v, f

    merged_wall_ids = []
    for i in range(len(quad_wall_mesh_lst)):
        if i in merged_wall_ids:
            continue

        mesh_0 = quad_wall_mesh_lst[i]
        normal_0 = mesh_0.vertex_normals[0, :]

        for j in range(i + 1, len(quad_wall_mesh_lst)):
            mesh_1 = quad_wall_mesh_lst[j]
            normal_1 = mesh_1.vertex_normals[0, :]
            # if two walls have common corners and normals are the same
            if np.allclose(np.abs(normal_0), np.abs(normal_1), atol=1e-2) and check_mesh_attachment(mesh_0, mesh_1):
                print(f'wall {i} and wall {j} are merged, normal: {normal_0}')

                # merge two wall
                mesh_vertices, mesh_faces = cat_mesh((mesh_0.vertices, mesh_0.faces), (mesh_1.vertices, mesh_1.faces))
                mesh_0 = trimesh.Trimesh(vertices=mesh_vertices,
                                         faces=mesh_faces,
                                         vertex_normals=np.tile(normal_0, (len(mesh_vertices), 1)))
                quad_wall_mesh_lst[i] = mesh_0
                merged_wall_ids.append(j)

    new_walls_lst = []
    for id in range(len(quad_wall_dict_lst)):
        if id in merged_wall_ids:
            continue

        mesh = quad_wall_mesh_lst[id]

        corners = np.array(mesh.vertices)
        if len(corners) > 4:
            # the wall is merged from multiple walls
            new_corners = []
            for i, c in enumerate(corners):
                dists = [np.linalg.norm(c - c1) for c1 in corners]
                dists[i] = np.inf
                print(f'corners dists: {dists}')
                if np.min(dists) < 1e-2:
                    continue
                new_corners.append(c)
            corners = np.array(new_corners)
        assert len(corners) == 4
        wall_size = mesh.bounding_box_oriented.extents

        wall_dict = {}
        wall_dict['ID'] = len(new_walls_lst)
        wall_dict['class'] = 'wall'
        # all the coordinates are in camera frame
        wall_dict['center'] = np.mean(corners, axis=0).tolist()
        wall_dict['normal'] = quad_wall_dict_lst[id]['normal']
        wall_dict['angles'] = quad_wall_dict_lst[id]['angles']
        wall_dict['width'] = wall_size[0] if wall_size[0] > 0 else wall_size[1]
        wall_dict['height'] = wall_size[2]
        wall_dict['corners'] = corners.tolist()

        new_walls_lst.append(wall_dict)
    return new_walls_lst


def parse_wall_corners(scene_annos: dict, room_id: str, camera_position_filepath: str):

    # read camera position file
    cam_pos_lst = []
    with open(camera_position_filepath) as f:
        cam_pos_lst = np.array([line.strip().split() for line in f if line.strip()], np.float32)
    assert len(cam_pos_lst) == 1, camera_position_filepath
    cam_position = cam_pos_lst[0]

    # parse corners
    junctions = np.array([item['coordinate'] for item in scene_annos['junctions']])
    lines_holes = []
    for semantic in scene_annos['semantics']:
        if semantic['type'] in ['window', 'door']:
            for planeID in semantic['planeID']:
                lines_holes.extend(np.where(np.array(scene_annos['planeLineMatrix'][planeID]))[0].tolist())

    lines_holes = np.unique(lines_holes)
    _, vertices_holes = np.where(np.array(scene_annos['lineJunctionMatrix'])[lines_holes])
    vertices_holes = np.unique(vertices_holes)

    # parse annotations
    walls = dict()
    walls_normal = dict()
    for semantic in scene_annos['semantics']:
        if semantic['ID'] != int(room_id):
            continue

        # find junctions of ceiling and floor
        for planeID in semantic['planeID']:
            plane_anno = scene_annos['planes'][planeID]

            if plane_anno['type'] != 'wall':
                lineIDs = np.where(np.array(scene_annos['planeLineMatrix'][planeID]))[0]
                lineIDs = np.setdiff1d(lineIDs, lines_holes)
                junction_pairs = [
                    np.where(np.array(scene_annos['lineJunctionMatrix'][lineID]))[0].tolist() for lineID in lineIDs
                ]
                wall = convert_lines_to_vertices(junction_pairs)
                walls[plane_anno['type']] = wall[0]

        # save normal of the vertical walls
        for planeID in semantic['planeID']:
            plane_anno = scene_annos['planes'][planeID]

            if plane_anno['type'] == 'wall':
                lineIDs = np.where(np.array(scene_annos['planeLineMatrix'][planeID]))[0]
                lineIDs = np.setdiff1d(lineIDs, lines_holes)
                junction_pairs = [
                    np.where(np.array(scene_annos['lineJunctionMatrix'][lineID]))[0].tolist() for lineID in lineIDs
                ]
                wall = convert_lines_to_vertices(junction_pairs)
                walls_normal[tuple(np.intersect1d(wall, walls['floor']))] = plane_anno['normal']

    # we assume that zs of floor equals 0, then the wall height is from the ceiling
    wall_height = np.mean(junctions[walls['ceiling']], axis=0)[-1]
    delta_height = np.array([0, 0, wall_height])

    # list of corner index
    wall_floor = walls['floor']

    # wall
    quad_wall_dict, quad_wall_normalized_dict = {}, {}
    quad_wall_lst, quad_wall_normalized_lst = [], []
    quad_wall_mesh_lst = []
    for i, j in zip(wall_floor, np.roll(wall_floor, shift=-1)):
        corner_i, corner_j = junctions[i], junctions[j]
        plane_normal = walls_normal[tuple(sorted([i, j]))]
        flip = verify_normal(corner_i, corner_j, delta_height, plane_normal)

        if flip:
            corner_j, corner_i = corner_i, corner_j

        # 3D coordinate for each wall
        quad_corners = np.array([corner_i, corner_i + delta_height, corner_j + delta_height, corner_j])
        # print(f'plane normal: {plane_normal}')
        wall_mesh = create_spatial_quad_polygen(quad_corners * 0.001,
                                                   plane_normal,
                                                   camera_center=cam_position * 0.001)
        quad_wall_mesh_lst.append(wall_mesh)

        wall_center = np.mean(quad_corners, axis=0)
        # wall center in camera frame, unit: meter
        wall_center_in_cam = (wall_center - cam_position) * 0.001
        wall_normal = np.array(plane_normal)
        # The direction of all camera is always along the positive y-axis.
        cos_angle = np.array(wall_normal).dot(np.array([0, -1, 0]))
        angle = np.arccos(cos_angle)
        if abs(cos_angle) < 1e-6:
            angle = np.pi / 2 if wall_normal[0] > 0 else -np.pi / 2
        wall_width = np.linalg.norm(corner_i - corner_j) * 0.001
        wall_height = np.linalg.norm(delta_height) * 0.001

        wall_dict = {}
        wall_dict['ID'] = len(quad_wall_lst)
        wall_dict['class'] = 'wall'
        # all the coordinates are in camera frame
        wall_dict['center'] = wall_center_in_cam.tolist()
        wall_dict['normal'] = wall_normal.tolist()
        wall_dict['angles'] = [0, 0, angle]
        wall_dict['width'] = wall_width
        wall_dict['height'] = wall_height
        wall_dict['corners'] = ((quad_corners - cam_position) * 0.001).tolist()
        quad_wall_lst.append(wall_dict)

    # quad_wall_lst = merge_walls(quad_wall_lst, quad_wall_mesh_lst)
    room_layout_mesh = trimesh.util.concatenate(quad_wall_mesh_lst)

    # compute the bounding box of the layout
    layout_bbox_min = trimesh.bounds.corners(room_layout_mesh.bounding_box_oriented.bounds).min(axis=0)
    layout_bbox_max = trimesh.bounds.corners(room_layout_mesh.bounding_box_oriented.bounds).max(axis=0)
    layout_bbox_size = layout_bbox_max - layout_bbox_min
    for wall_dict in quad_wall_lst:
        wall_center_in_cam = np.array(wall_dict['center'])
        wall_angles = wall_dict['angles']
        # angle = np.arcsin(wall_angles[1]) if abs(wall_angles[0]) < 5e-3 else np.arccos(wall_angles[0])

        wall_corners = np.array(wall_dict['corners'])
        wall_corners_normalized = wall_corners / layout_bbox_size
        wall_width_normalized = np.linalg.norm(wall_corners_normalized[0] - wall_corners_normalized[3])
        wall_height_normalized = float(wall_dict['height']) / layout_bbox_size[2]

        wall_normalized_dict = {}
        wall_normalized_dict['ID'] = wall_dict['ID']
        wall_normalized_dict['class'] = wall_dict['class']
        wall_normalized_dict['center'] = (wall_center_in_cam / layout_bbox_size).tolist()
        wall_normalized_dict['normal'] = wall_dict['normal']
        wall_normalized_dict['angles'] = [np.cos(wall_angles[2]), np.sin(wall_angles[2])]
        # the width could be weird if the layout is nor square
        wall_normalized_dict['width'] = wall_width_normalized
        wall_normalized_dict['height'] = wall_height_normalized
        quad_wall_normalized_lst.append(wall_normalized_dict)

    quad_wall_dict['walls'] = quad_wall_lst
    quad_wall_normalized_dict['walls'] = quad_wall_normalized_lst
    return quad_wall_dict, quad_wall_normalized_dict, room_layout_mesh


from src.diffu_layout.improved_diffusion.clip_util import FrozenCLIPEmbedder


def save_visualization_and_mesh(
    objects_lst: List[Dict],
    quad_walls_lst: List[Dict],
    source_cor_path: str,
    source_img_path: str,
):
    """ get visualization of the layout and mesh

    Args:
        objects_lst (List[Dict]): objects bbox list
        quad_walls_lst (List[Dict]): quad walls list
        source_cor_path (str): 2d wall corners filepath
        source_img_path (str): source panorama filepath

    Returns:
        _type_: semantic image and mesh
    """
    # visualize quad wall if exists
    assert len(objects_lst)
    assert len(quad_walls_lst)

    quad_walls_lst_cp = copy.deepcopy(quad_walls_lst)
    objects_lst_cp = copy.deepcopy(objects_lst)
    for quad_wall in quad_walls_lst_cp:
        wall_dict = {}

        wall_dict['center'] = quad_wall['center']
        wall_dict['size'] = [quad_wall['width'], 0.01, quad_wall['height']]
        normal = quad_wall['normal']
        # The direction of all camera is always along the negative y-axis.
        cos_angle = np.array(normal).dot(np.array([0, -1, 0]))
        angle = np.arccos(cos_angle)
        if abs(cos_angle) < 1e-6:
            angle = np.pi / 2 if normal[0] > 0 else -np.pi / 2

        wall_dict['angles'] = [0, 0, angle]
        wall_dict['class'] = 'wall'
        objects_lst_cp.append(wall_dict)

    # visualize room layout and object bbox in panorama
    # rgb_img = cv2.imread(source_img_path, cv2.IMREAD_UNCHANGED)
    # rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    # bg_img = rgb_img
    img_H, img_W = 512, 1024
    bg_img = np.zeros((img_H, img_W, 3), dtype=np.uint8)
    # Read ground truth corners
    with open(source_cor_path, 'r') as f:
        corners_lst = np.array([line.strip().split() for line in f if line.strip()], np.float32)
        # Corner with minimum x should at the beginning
        corners_lst = np.roll(corners_lst[:, :2], -2 * np.argmin(corners_lst[::2, 0]), 0)
        # Detect occlusion
        occlusion = find_occlusion(corners_lst[::2].copy()).repeat(2)
        # corners correspondenses' x coordinate should be identical
        assert (np.abs(corners_lst[0::2, 0] - corners_lst[1::2, 0]) > img_W / 100).sum() == 0, source_img_path
        # corners correspondenses' y coordinate should be y_floor < y_ceiling
        assert (corners_lst[0::2, 1] > corners_lst[1::2, 1]).sum() == 0, source_img_path
    # 1d ceiling-wall/floor-wall boundary, [-pi/2, pi/2]
    boundary_lst = corners_to_1d_boundary(corners_lst, img_H, img_W)
    sem_bbox_img = vis_floor_ceiling(image=bg_img, coords_2d=boundary_lst, color_to_labels=COLOR_TO_ADEK_LABEL)
    sem_bbox_img = vis_objs3d(image=bg_img,
                              v_bbox3d=objects_lst_cp,
                              camera_position=np.array([0, 0, 0]),
                              color_to_labels=COLOR_TO_ADEK_LABEL,
                              b_show_axes=False,
                              b_show_centroid=False,
                              b_show_bbox3d=False,
                              b_show_info=False,
                              b_show_polygen=True,
                              thickness=2)
    # convert BGR to RGB
    debug_bbox_img = sem_bbox_img

    debug_bbox_trimesh = vis_scene_mesh(room_layout_mesh=None,
                                        obj_bbox_lst=objects_lst_cp,
                                        color_to_labels=COLOR_TO_ADEK_LABEL,
                                        room_layout_bbox=None)

    return debug_bbox_img, debug_bbox_trimesh


def prepare_dataset(raw_dataset_dir: str,
                    target_room_type: str,
                    scene_ids: List,
                    out_dir: str,
                    annotated_labels_dir: str,
                    split: str = 'train',
                    b_save_debug_files=False):
    """
    Prepare the dataset for training and testing.
    :param raw_dataset_dir: str, the directory of the raw dataset.
    :param target_room_type: str, the target room type.
    :param scene_ids: List, the list of scene ids.
    :param out_dir: str, the output directory.
    :param annotated_labels_dir: str, the directory of the annotated labels.
    :param b_save_debug_files: bool, whether to save debug files.
    """

    furniture_counts = []

    room_layout_size_lst = []

    quad_wall_num_lst = []

    # clip model
    glove_model = FrozenCLIPEmbedder(device='cuda')

    st3d_rooms_lst = []
    for scene_id in tqdm(scene_ids):

        if scene_id in INVALID_SCENES_LST:
            continue

        room_type_lst = None
        # parse scene annotation
        scene_anno_3d_filepath = os.path.join(raw_dataset_dir, scene_id, 'annotation_3d.json')
        if not os.path.isfile(scene_anno_3d_filepath):
            INVALID_SCENES_LST.append(scene_id)
            continue
        else:
            scene_anno_3d_dict = json.load(open(scene_anno_3d_filepath, 'r'))
            room_type_lst = scene_anno_3d_dict['semantics']

        scene_dir = os.path.join(raw_dataset_dir, scene_id, '2D_rendering')
        for room_id in np.sort(os.listdir(scene_dir)):

            room_str = '%s_%s' % (scene_id, room_id)
            if room_str in INVALID_ROOMS_LST:
                continue
            room_type_str = 'undefined'
            if room_type_lst is not None:
                for rt in room_type_lst:
                    if rt['ID'] == int(room_id):
                        room_type_str = rt['type']
                        break
            if room_type_str != target_room_type:
                continue

            # print(f'Processing room: {room_str}')

            room_path = os.path.join(scene_dir, room_id, "panorama")
            source_img_path = os.path.join(room_path, "full", "rgb_rawlight.png")
            source_cor_path = os.path.join(room_path, "layout.txt")
            source_cam_pos_path = os.path.join(room_path, "camera_xyz.txt")
            source_depth_path = os.path.join(room_path, "full", "depth.png")

            # parse room layout
            # room_layout_mesh = parse_room_layout(source_img_path, source_cam_pos_path, source_cor_path)
            quad_walls_dict, quad_walls_normalized_dict, room_layout_mesh = parse_wall_corners(
                scene_anno_3d_dict, room_id, source_cam_pos_path)
            # skip wall number < 4
            if len(quad_walls_normalized_dict['walls']) < 4:
                print(f'bad scene {room_str} walls number < 4')
                INVALID_ROOMS_LST.append(room_str)
                continue
            # skip wall height > 5m
            if np.max([wall['height'] for wall in quad_walls_dict['walls']]) > 5.0:
                print(f'bad scene {room_str} walls height > 5m')
                INVALID_ROOMS_LST.append(room_str)
                continue
            quad_wall_num_lst.append(len(quad_walls_normalized_dict['walls']))
            if target_room_type in ['bedroom', 'kitchen', 'study', 'bathroom']:
                if len(quad_walls_normalized_dict['walls']) > ST3D_BEDROOM_QUAD_WALL_MAX_LEN:
                    print(f'bad scene {room_str} walls number > {ST3D_BEDROOM_QUAD_WALL_MAX_LEN}')
                    ROOM_WALLS_LARGER_THAN_10.append(room_str)
                    continue
            elif target_room_type == 'living room':
                if len(quad_walls_normalized_dict['walls']) > ST3D_LIVINGROOM_QUAD_WALL_MAX_LEN:
                    print(f'bad scene {room_str} walls number > {ST3D_LIVINGROOM_QUAD_WALL_MAX_LEN}')
                    ROOM_WALLS_LARGER_THAN_10.append(room_str)
                    continue

            # parse 3d bbox of objects in the room
            new_labeld_room_filepath = os.path.join(annotated_labels_dir, room_str + '.json')
            if not os.path.exists(new_labeld_room_filepath):
                # new_labeld_room_filepath = None
                continue
            obj_bbox_3d_dict, obj_bbox_3d_normalized_dict = parse_bbox_in_room(room_path, room_layout_mesh,
                                                                               new_labeld_room_filepath)
            if obj_bbox_3d_dict is None:
                print(f'bad scene {room_str} objects number < {ST3D_LIVINGROOM_MIN_LEN}')
                INVALID_ROOMS_LST.append(room_str)
                continue

            # visialization and debug
            if b_save_debug_files:
                debug_bbox_img, debug_bbox_trimesh = save_visualization_and_mesh(
                    objects_lst=obj_bbox_3d_dict['objects'],
                    quad_walls_lst=quad_walls_dict['walls'],
                    source_cor_path=source_cor_path,
                    source_img_path=source_img_path,
                )

            # generate scene description
            scene_desc_text, scene_desc_emb = get_scene_description(
                room_type=room_type_str,
                wall_dict=quad_walls_dict.copy(),
                object_dict=obj_bbox_3d_dict.copy(),
                glove_model=glove_model,
                eval=(split == 'test'),
            )
            # print(f'room {room_str} scene_desc_text: {scene_desc_text}')
            
            out_img_dir = os.path.join(out_dir, room_type_str.replace(' ', ''), 'img')
            out_depth_dir = os.path.join(out_dir, room_type_str.replace(' ', ''), 'depth')
            out_cord_dir = os.path.join(out_dir, room_type_str.replace(' ', ''), 'label_cor')
            out_cam_pos_dir = os.path.join(out_dir, room_type_str.replace(' ', ''), 'cam_pos')
            out_room_type_dir = os.path.join(out_dir, room_type_str.replace(' ', ''), 'room_type')
            out_bbox_3d_dir = os.path.join(out_dir, room_type_str.replace(' ', ''), 'bbox_3d')
            out_quad_wall_dir = os.path.join(out_dir, room_type_str.replace(' ', ''), 'quad_walls')
            out_text_desc_dir = os.path.join(out_dir, room_type_str.replace(' ', ''), 'text_desc')
            out_text_emb_dir = os.path.join(out_dir, room_type_str.replace(' ', ''), 'text_desc_emb')
            out_sem_bbox_img_dir = os.path.join(out_dir, room_type_str.replace(' ', ''), 'sem_bbox_img')
            out_sem_layout_img_dir = os.path.join(out_dir, room_type_str.replace(' ', ''), 'sem_layout_img')
            os.makedirs(out_img_dir, exist_ok=True)
            os.makedirs(out_depth_dir, exist_ok=True)
            os.makedirs(out_cord_dir, exist_ok=True)
            os.makedirs(out_cam_pos_dir, exist_ok=True)
            os.makedirs(out_room_type_dir, exist_ok=True)
            os.makedirs(out_bbox_3d_dir, exist_ok=True)
            os.makedirs(out_quad_wall_dir, exist_ok=True)
            os.makedirs(out_text_desc_dir, exist_ok=True)
            os.makedirs(out_text_emb_dir, exist_ok=True)
            os.makedirs(out_sem_bbox_img_dir, exist_ok=True)
            os.makedirs(out_sem_layout_img_dir, exist_ok=True)
            target_img_path = os.path.join(out_img_dir, '%s_%s.png' % (scene_id, room_id))
            target_depth_path = os.path.join(out_depth_dir, '%s_%s.png' % (scene_id, room_id))
            target_cor_path = os.path.join(out_cord_dir, '%s_%s.txt' % (scene_id, room_id))
            target_cam_pos_path = os.path.join(out_cam_pos_dir, '%s_%s.txt' % (scene_id, room_id))
            target_room_type_path = os.path.join(out_room_type_dir, '%s_%s.txt' % (scene_id, room_id))
            target_bbox_3d_path = os.path.join(out_bbox_3d_dir, '%s_%s.json' % (scene_id, room_id))
            target_bbox_3d_normal_path = os.path.join(out_bbox_3d_dir, '%s_%s_normalized.json' % (scene_id, room_id))
            target_sem_bbox_img_path = os.path.join(out_sem_bbox_img_dir, '%s_%s.png' % (scene_id, room_id))
            target_sem_layout_img_path = os.path.join(out_sem_layout_img_dir, '%s_%s.png' % (scene_id, room_id))
            target_bbox_3d_mesh_path = os.path.join(out_bbox_3d_dir, '%s_%s.ply' % (scene_id, room_id))
            target_quad_wall_path = os.path.join(out_quad_wall_dir, '%s_%s.json' % (scene_id, room_id))
            target_quad_wall_normalized_path = os.path.join(out_quad_wall_dir,
                                                            '%s_%s_normalized.json' % (scene_id, room_id))
            target_text_desc_path = os.path.join(out_text_desc_dir, '%s_%s.txt' % (scene_id, room_id))
            target_text_emb_path = os.path.join(out_text_emb_dir, '%s_%s.npy' % (scene_id, room_id))

            # skip rooms without bed
            if target_room_type == 'bedroom':
                room_furniture_types = set([box['class'] for box in obj_bbox_3d_dict['objects']])
                if 'bed' not in room_furniture_types:
                    INVALID_ROOMS_LST.append(room_str)
                    print(f'bad scene {room_str} without bed')
                    continue
            # skip kitchen without stove
            if target_room_type == 'kitchen':
                room_furniture_types = set([box['class'] for box in obj_bbox_3d_dict['objects']])
                if 'stove' not in room_furniture_types:
                    INVALID_ROOMS_LST.append(room_str)
                    print(f'bad kitchen {room_str} without stove')
                    continue

            # skip rooms with corrupted files
            if not os.path.isfile(source_img_path) or not os.path.isfile(source_depth_path) or not os.path.isfile(source_cor_path) \
            or not os.path.isfile(source_cam_pos_path) or imghdr.what(source_img_path) is None:
                INVALID_ROOMS_LST.append(room_str)
                print(f'bad scene {room_str} with corrupted files')
                continue
            else:
                # construct st3d room
                st3d_room = St3dRoom(scene_id=room_str, scene_type=room_type_str,
                                walls_lst=quad_walls_dict['walls'],
                                bboxes_lst=obj_bbox_3d_dict['objects'],)
                st3d_rooms_lst.append(st3d_room)
                print(f'scene_id: {room_str}, room_centroid: {st3d_room.centroid}')
                # re-center the walls and objects
                def recenter_fn(x:Dict)->Dict:
                    x['center'] = (np.array(x['center']) - st3d_room.centroid).tolist()
                    x['corners'] = [(np.array(cor) - st3d_room.centroid).tolist() for cor in x['corners']]
                    return x
                quad_walls_dict['walls'] = list(map(recenter_fn, quad_walls_dict['walls']))
                obj_bbox_3d_dict['objects'] = list(map(recenter_fn, obj_bbox_3d_dict['objects']))
                    
                shutil.copyfile(source_img_path, target_img_path)
                shutil.copyfile(source_depth_path, target_depth_path)
                shutil.copyfile(source_cor_path, target_cor_path)
                shutil.copyfile(source_cam_pos_path, target_cam_pos_path)
                # write room type
                with open(target_room_type_path, 'w') as f:
                    f.write(room_type_str)
                # write 3d bbox
                with open(target_bbox_3d_path, 'w') as f:
                    json.dump(obj_bbox_3d_dict, f, indent=4)
                # write quad walls
                with open(target_quad_wall_path, 'w') as f:
                    json.dump(quad_walls_dict, f, indent=4)

                # write text description
                with open(target_text_desc_path, 'w') as f:
                    f.write(scene_desc_text)
                # write text embedding
                np.save(target_text_emb_path, scene_desc_emb)

                if b_save_debug_files:
                    # visualize semantic layout img
                    Image.fromarray(debug_bbox_img).save(target_sem_layout_img_path)

                    debug_bbox_trimesh.export(target_bbox_3d_mesh_path)

            # furniture statistics
            if target_room_type == 'bedroom':
                room_furniture_types = set([box['class'] for box in obj_bbox_3d_dict['objects']])
                ST3D_BEDROOM_FURNITURES_SET.update(room_furniture_types)
            elif target_room_type == 'living room':
                room_furniture_types = set([box['class'] for box in obj_bbox_3d_dict['objects']])
                ST3D_LIVINGROOM_FURNITURES_SET.update(room_furniture_types)
            elif target_room_type == 'dining room':
                room_furniture_types = set([box['class'] for box in obj_bbox_3d_dict['objects']])
                ST3D_DININGROOM_FURNITURES_SET.update(room_furniture_types)
            elif target_room_type == 'kitchen':
                room_furniture_types = set([box['class'] for box in obj_bbox_3d_dict['objects']])
                ST3D_KITCHEN_FURNITURES_SET.update(room_furniture_types)
            elif target_room_type == 'study':
                room_furniture_types = set([box['class'] for box in obj_bbox_3d_dict['objects']])
                ST3D_STUDY_FURNITURES_SET.update(room_furniture_types)
            elif target_room_type == 'bathroom':
                room_furniture_types = set([box['class'] for box in obj_bbox_3d_dict['objects']])
                ST3D_BATHROOM_FURNITURES_SET.update(room_furniture_types)

            room_layout_size = room_layout_mesh.bounding_box_oriented.extents
            room_layout_size_lst.append(room_layout_size)
            furniture_counts.append([box['class'] for box in obj_bbox_3d_dict['objects']])

    st3d_dataset = ST3DDataset.from_dataset_directory(st3d_rooms_lst)
    tr_bounds = st3d_dataset.bounds["translations"]
    si_bounds = st3d_dataset.bounds["sizes"]
    an_bounds = st3d_dataset.bounds["angles"]

    dataset_stats = {
        "bounds_translations": tr_bounds[0].tolist() + tr_bounds[1].tolist(),
        "bounds_sizes": si_bounds[0].tolist() + si_bounds[1].tolist(),
        "bounds_angles": an_bounds[0].tolist() + an_bounds[1].tolist(),
        "class_labels": st3d_dataset.class_labels,
        "object_types": st3d_dataset.object_types,
        "class_frequencies": st3d_dataset.class_frequencies,
        "class_order": st3d_dataset.class_order,
        "count_furniture": st3d_dataset.count_furniture,
        "room_layout_size_mean": np.mean(room_layout_size_lst, axis=0).tolist(),
    }
    print(dataset_stats)
    dataset_stat_filepath = os.path.join(out_dir, target_room_type.replace(' ', ''), split+"_dataset_stats.json")
    with open(dataset_stat_filepath, "w") as f:
        json.dump(dataset_stats, f, indent=4)
        
    furniture_counts = Counter(sum(furniture_counts, []))
    furniture_counts = OrderedDict(sorted(furniture_counts.items(), key=lambda x: -x[1]))
    room_mean_size = np.mean(np.array(room_layout_size_lst), axis=0)
    quad_wall_num_max = max(quad_wall_num_lst)
    quad_wall_num_mean = np.mean(np.array(quad_wall_num_lst), axis=0)
    print(f"furniture_counts: \n {furniture_counts}")
    print(f'mean room_layout size : {room_mean_size}')
    print(f'max quad wall num: {quad_wall_num_max}')
    print(f'mean quad wall num: {quad_wall_num_mean}')

    return furniture_counts, room_mean_size, quad_wall_num_max, quad_wall_num_mean


def parse_args():
    parser = argparse.ArgumentParser(description="Structured3D 2D Layout Visualization")
    parser.add_argument("--dataset_path",
                        default="/data/dataset/Structured3D/Structured3D/",
                        help="raw dataset path",
                        metavar="DIR")
    parser.add_argument("--room_type",
                        default="st3d_bathroom",
                        choices=["st3d_bedroom", "st3d_livingroom", "st3d_diningroom", "st3d_study", "st3d_kitchen", "st3d_bathroom"],
                        help="structured3d room type")
    parser.add_argument('--annotated_labels_path',
                        type=str,
                        default='/data/dataset/Structured3D/preprocessed/annotations/bathroom/latest_labels/',
                        help='path to annotated labels')
    parser.add_argument('--out_train_path',
                        default='/mnt/nas_3dv/hdd1/datasets/Structured3d/preprocessed/20240219_text2pano/train')
    parser.add_argument('--out_test_path',
                        default='/mnt/nas_3dv/hdd1/datasets/Structured3d/preprocessed/20240219_text2pano/test')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.room_type == 'st3d_bedroom':
        room_type_str = 'bedroom'
    elif args.room_type == 'st3d_livingroom':
        room_type_str = 'living room'
    elif args.room_type == 'st3d_diningroom':
        room_type_str = 'dining room'
    elif args.room_type == 'st3d_study':
        room_type_str = 'study'
    elif args.room_type == 'st3d_kitchen':
        room_type_str = 'kitchen'
    elif args.room_type == 'st3d_bathroom':
        room_type_str = 'bathroom'
    else:
        raise ValueError(f'unsupported room type {args.room_type}')

    train_furniture_stats, room_mean_size, wall_num_max, wall_num_mean = prepare_dataset(args.dataset_path,
                                                                                         room_type_str,
                                                                                         TRAIN_SCENE,
                                                                                         args.out_train_path,
                                                                                         args.annotated_labels_path,
                                                                                         split='train',
                                                                                         b_save_debug_files=True)
    test_furniture_stats, _, _, _ = prepare_dataset(args.dataset_path,
                                                    room_type_str,
                                                    TEST_SCENE,
                                                    args.out_test_path,
                                                    args.annotated_labels_path,
                                                    split='test',
                                                    b_save_debug_files=True)

    if args.room_type == 'st3d_bedroom':
        print('*' * 20 + ' bedroom furniture types: ' + '*' * 20)
        print(ST3D_BEDROOM_FURNITURES_SET)
    elif args.room_type == 'st3d_livingroom':
        print('*' * 20 + ' livingroom furniture types: ' + '*' * 20)
        print(ST3D_LIVINGROOM_FURNITURES_SET)
    elif args.room_type == 'st3d_diningroom':
        print('*' * 20 + ' st3d_diningroom furniture types: ' + '*' * 20)
        print(ST3D_DININGROOM_FURNITURES_SET)
    elif args.room_type == 'st3d_kitchen':
        print('*' * 20 + ' st3d_kitchen furniture types: ' + '*' * 20)
        print(ST3D_KITCHEN_FURNITURES_SET)
    elif args.room_type == 'st3d_study':
        print('*' * 20 + ' st3d_study furniture types: ' + '*' * 20)
        print(ST3D_STUDY_FURNITURES_SET)
    elif args.room_type == 'st3d_bathroom':
        print('*' * 20 + ' st3d_bathroom furniture types: ' + '*' * 20)
        print(ST3D_BATHROOM_FURNITURES_SET)

    # merge train and test furniture statistics
    for k, v in train_furniture_stats.items():
        if k in test_furniture_stats:
            v += test_furniture_stats[k]
        train_furniture_stats[k] = v

    dataset_stats = {
        'invalid_rooms': INVALID_ROOMS_LST,
        'bedroom_furniture_types': list(ST3D_BEDROOM_FURNITURES_SET),
        'livingroom_furniture_types': list(ST3D_LIVINGROOM_FURNITURES_SET),
        'diningroom_furniture_types': list(ST3D_DININGROOM_FURNITURES_SET),
        'kitchen_furniture_types': list(ST3D_KITCHEN_FURNITURES_SET),
        'study_furniture_types': list(ST3D_STUDY_FURNITURES_SET),
        'bathroom_furniture_types': list(ST3D_BATHROOM_FURNITURES_SET),
        'furniture_counter': train_furniture_stats,
        'room_mean_size': room_mean_size.tolist(),
        'quad_wall_num_max': wall_num_max,
        'quad_wall_num_mean': wall_num_mean,
    }
    dataset_stats_filepath = os.path.join(os.path.dirname(args.out_train_path),
                                          room_type_str.replace(' ', '_') + '_dataset_stats.json')
    with open(dataset_stats_filepath, 'w') as f:
        json.dump(dataset_stats, f, indent=4)

    # get train.json and test.json
    train_test_split_filepath = os.path.join(os.path.dirname(args.out_test_path), room_type_str.replace(' ', '')+'_splits.json')
    valid_train_scene_ids = [
        s[:-4] for s in os.listdir(os.path.join(args.out_train_path, room_type_str.replace(' ', ''), 'img'))
    ]
    valid_test_scene_ids = [
        s[:-4] for s in os.listdir(os.path.join(args.out_test_path, room_type_str.replace(' ', ''), 'img'))
    ]

    with open(train_test_split_filepath, 'w') as f:
        for i, scene_id in enumerate(valid_train_scene_ids):
            f.write(scene_id + ',' + 'train' + '\n')
        for i, scene_id in enumerate(valid_test_scene_ids):
            f.write(scene_id + ',' + 'test' + '\n')


if __name__ == "__main__":
    main()
