#
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
#
"""Script used for parsing the 3D-FRONT data scenes into numpy files in order
to be able to avoid I/O overhead when training our model.
"""
import argparse
import logging
import json
import os
import sys

sys.path.append('.')
sys.path.append('..')

import copy

import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm
from typing import List, Dict

import open3d as o3d
import trimesh
import cv2

# from utils import DirLock, ensure_parent_directory_exists, \
#     floor_plan_renderable, floor_plan_from_scene, \
#     get_textured_objects_in_scene, scene_from_args, render

from dataset.threed_front import filter_function

from dataset.threed_front.threed_front import ThreedFront
from dataset.threed_front.threed_front_dataset import dataset_encoding_factory
from dataset.threed_front.threed_front_scene import Room
from dataset.gen_scene_text import get_scene_description

from utils.equirect_projection import vis_objs3d, vis_floor_ceiling_simple
from utils.utils import euler_angle_to_matrix, matrix_to_euler_angles
from src.diffu_layout.improved_diffusion.clip_util import FrozenCLIPEmbedder
from dataset.threed_front.metadata import TDFRONT_COLOR_TO_ADEK_LABEL

ALL_SCENE = ['train', 'val', 'test']
TRAIN_SCENE = ['train', 'val']
TEST_SCENE = ['test']


def vis_scene_mesh(room_layout_mesh: trimesh.Trimesh,
                   obj_bbox_lst: List[Dict],
                   color_to_labels: Dict = None,
                   heading_axis: int = 2,
                   room_layout_bbox=None) -> trimesh.Trimesh:

    def create_oriented_bbox(scene_bbox: List[Dict]) -> trimesh.Trimesh:
        """Export oriented (around Y axis) scene bbox to meshes
        Args:
            scene_bbox: (N x 7 numpy array): xyz pos of center and 3 lengths (dx,dy,dz)
                and heading angle around Z axis.
                Y forward, X right, Z upward. heading angle of positive X is 0,
                heading angle of positive Y is 90 degrees.
            out_filename: (string) filename
        """

        def heading2rotmat(heading_angle_rad):
            rotmat = np.eye(3)
            cosval = np.cos(heading_angle_rad)
            sinval = np.sin(heading_angle_rad)
            if heading_axis == 2:
                # rotate around Z axis
                rotmat[0:2, 0:2] = np.array([[cosval, -sinval], [sinval, cosval]])
            elif heading_axis == 1:
                rotmat[0, 0] = cosval
                rotmat[0, 2] = -sinval
                rotmat[2, 0] = sinval
                rotmat[2, 2] = cosval
            else:
                raise NotImplementedError
            return rotmat

        def convert_oriented_box_to_trimesh_fmt(box, color_to_labels: Dict = None):
            box_center = box['center']
            box_lengths = box['size']
            transform_matrix = np.eye(4)
            transform_matrix[0:3, 3] = box_center
            # only use Y angle, rad
            transform_matrix[0:3, 0:3] = heading2rotmat(box['angles'][heading_axis])
            box_trimesh_fmt = trimesh.creation.box(box_lengths, transform_matrix)
            if color_to_labels is not None:
                labels_lst = list(color_to_labels.values())
                colors_lst = list(color_to_labels.keys())
                color = colors_lst[labels_lst.index(box['class'])]
            else:
                color = (np.random.random(3) * 255).astype(np.uint8).tolist()
            box_trimesh_fmt.visual.face_colors = color
            return box_trimesh_fmt

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


def create_spatial_quad_polygen(quad_vertices: np.array, normal: np.array, camera_center: np.array):
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

    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    pcd_o3d.points.append(normal_point)
    pcd_o3d.points.append(centroid)

    # pcd = trimesh.PointCloud(np.asarray(pcd.points))
    return mesh, pcd_o3d


def parse_room_layout(room: Room, all_class_labels: List, R: np.array = np.eye(3)):
    """ parse room layout: quad walls, 

    Args:
        room (Room): room instance
        all_class_labels (List): all class labels of current room type
        R (np.array, optional): rotation matrix used to otate original coordinate system to x-right, y-forward, z-up
                                . Defaults to np.eye(3).

    Returns:
        Dict: quad walls dict.
    """
    scene_bbox_size = room.scene_bbox_size
    scene_bbox_size = np.array([scene_bbox_size[0], scene_bbox_size[2], scene_bbox_size[1]])
    scene_bbox_center = room.scene_bbox_centroid

    # original coordinate system in 3D-Front is x-right, y-up, z-forward
    quad_wall_model_lst = [q for q in room.extras if q.model_type == "wall"]
    quad_wall_dict_lst = room.quad_walls

    # the coordinate we need is x-right, y-forward, z-up
    new_quad_wall_dict_lst = []
    new_quad_wall_normalized_dict_lst = []

    for wall, wall_dict in zip(quad_wall_model_lst, quad_wall_dict_lst):

        old_wall_dict = wall_dict.copy()
        new_wall_dict = wall_dict.copy()

        wall_center = np.array(wall_dict['center']) - scene_bbox_center
        wall_normal = np.array(wall_dict['normal'])
        wall_width = float(wall_dict['width'])
        wall_height = float(wall_dict['height'])
        wall_corners = np.array(wall_dict['corners']) - scene_bbox_center
        original_angle = wall.z_angle
        if np.isnan(original_angle):
            print(f'room {room.uid} wall {wall_dict["ID"]} angle is nan, wall normal: {wall_normal}')
            continue

        old_wall_dict['center'] = wall_center
        old_wall_dict['normal'] = wall_normal
        old_wall_dict['size'] = [wall_width, wall_height, 0.01]
        old_wall_dict['angles'] = [0, original_angle, 0]

        new_center = R @ wall_center
        new_normal = R @ wall_normal
        new_corners = (R @ (wall_corners).T).T
        # The forward direction is negative y-axis.
        cos_angle = np.array(new_normal).dot(np.array([0, -1, 0]))
        new_angle = np.arccos(cos_angle)
        if abs(cos_angle) < 1e-6:
            new_angle = np.pi / 2 if new_normal[0] > 0 else -np.pi / 2
        new_wall_dict['angles'] = [0, 0, new_angle]
        new_wall_dict['center'] = new_center
        new_wall_dict['normal'] = new_normal
        new_wall_width = np.linalg.norm(new_corners[3] - new_corners[0])
        new_wall_height = np.linalg.norm(new_corners[1] - new_corners[0])
        new_wall_dict['size'] = [new_wall_width, 0.01, new_wall_height]
        new_wall_dict['corners'] = new_corners

        # if room.uid == '01805656-e66f-44b1-8bc1-5e722fff3fff_Bedroom-8715':
        #     wall_vertices = (R @ (np.array(wall_dict['corners']) - scene_bbox_center).T).T
        #     _, wall_pcd = create_spatial_quad_polygen(wall_vertices, new_normal, camera_center=None)
        #     o3d.io.write_point_cloud(
        #         f'/mnt/nas_3dv/hdd1/datasets/3D_FRONT_FUTURE/01805656-e66f-44b1-8bc1-5e722fff3fff_Bedroom-8715/quad_wall_{wall_dict["ID"]}.ply',
        #         wall_pcd)

        # if layout_bbox_size is not None:
        wall_normalized_dict = {}
        wall_normalized_dict['ID'] = wall_dict['ID']
        # wall_normalized_dict['class'] = wall.one_hot_label(all_class_labels).tolist()
        wall_normalized_dict['class'] = 'wall'
        wall_normalized_dict['center'] = (new_center / scene_bbox_size).tolist()
        wall_normalized_dict['normal'] = new_normal.tolist()
        wall_normalized_dict['angles'] = [np.cos(new_angle), np.sin(new_angle)]
        # wall_normalized_dict['angles'] = [0, 0, new_angle]
        # wall_normalized_dict['width'] = (wall_width / max(scene_bbox_size[0], scene_bbox_size[1]))
        # wall_normalized_dict['height'] = (wall_height / scene_bbox_size[2])
        normalized_corners = new_corners / scene_bbox_size
        wall_normalized_dict['width'] = np.linalg.norm(normalized_corners[3] - normalized_corners[0])
        wall_normalized_dict['height'] = np.linalg.norm(normalized_corners[1] - normalized_corners[0])
        wall_normalized_dict['size'] = [wall_normalized_dict['width'], 0.01, wall_normalized_dict['height']]

        new_quad_wall_dict_lst.append(new_wall_dict)
        new_quad_wall_normalized_dict_lst.append(wall_normalized_dict)

    return {'walls': new_quad_wall_dict_lst}, {'walls': new_quad_wall_normalized_dict_lst}


def parse_bbox_in_room(room: Room, all_class_labels: List, R: np.array = np.eye(3)):
    """ parse furniture in the room: doors, windows, objects

    Args:
        room (Room): room instance
        all_class_labels (List): all class labels of current room type
        R (np.array, optional): rotation matrix used to otate original coordinate system to x-right, y-forward, z-up
                                . Defaults to np.eye(3).

    Returns:
        Dict: objects dict.
    """
    scene_bbox_size = room.scene_bbox_size
    scene_bbox_size = np.array([scene_bbox_size[0], scene_bbox_size[2], scene_bbox_size[1]])
    scene_bbox_center = room.scene_bbox_centroid

    doors_dict_lst = room.doors
    door_model_lst = [d for d in room.extras if d.model_type == "door"]
    windows_dict_lst = room.windows
    window_model_lst = [w for w in room.extras if w.model_type == "window"]
    object_lst = room.bboxes

    obj_dict_lst, obj_normalized_dict_lst = [], []

    for door_dict, door in zip(doors_dict_lst, door_model_lst):
        old_door_dict = door_dict.copy()
        new_door_dict = door_dict.copy()

        door_center = door.centroid() - scene_bbox_center
        door_size = door.size
        original_angle = door.z_angle
        if np.isnan(original_angle):
            print(f'room {room.uid} door {door_dict["ID"]} angle is nan')
            continue

        old_door_dict['center'] = door_center
        old_door_dict['size'] = door_size
        old_door_dict['angles'] = [0, original_angle, 0]

        new_center = R @ door_center
        new_normal = R @ door_dict['normal']
        new_size = [door_size[0], door_size[2], door_size[1]]
        # The forward direction  is negative y-axis.
        cos_angle = np.array(new_normal).dot(np.array([0, -1, 0]))
        new_angle = np.arccos(cos_angle)
        if abs(cos_angle) < 1e-6:
            new_angle = np.pi / 2 if new_normal[0] > 0 else -np.pi / 2

        new_door_dict['center'] = new_center
        new_door_dict['normal'] = new_normal
        new_door_dict['size'] = new_size
        new_door_dict['angles'] = [0, 0, new_angle]

        door_normalized_dict = {}
        door_normalized_dict['ID'] = new_door_dict['ID']
        # door_normalized_dict['class'] = door.one_hot_label(all_class_labels).tolist()
        door_normalized_dict['class'] = 'door'
        door_normalized_dict['center'] = (new_center / scene_bbox_size).tolist()
        door_normalized_dict['angles'] = [np.cos(new_angle), np.sin(new_angle)]
        door_normalized_dict['size'] = (new_size / scene_bbox_size).tolist()

        obj_dict_lst.append(new_door_dict)
        obj_normalized_dict_lst.append(door_normalized_dict)

    for window_dict, window in zip(windows_dict_lst, window_model_lst):
        old_window_dict = window_dict.copy()
        new_window_dict = window_dict.copy()

        window_center = window.centroid() - scene_bbox_center
        window_size = window.size
        original_angle = window.z_angle
        if np.isnan(original_angle):
            print(f'room {room.uid} window {window_dict["ID"]} angle is nan')
            continue

        old_window_dict['center'] = window_center
        old_window_dict['size'] = window_size
        old_window_dict['angles'] = [0, original_angle, 0]

        new_center = R @ window_center
        new_normal = R @ window_dict['normal']
        new_size = [window_size[0], window_size[2], window_size[1]]
        # we define the forward direction is negative y-axis.
        cos_angle = new_normal.dot(np.array([0, -1, 0]))
        new_angle = np.arccos(cos_angle)
        if abs(cos_angle) < 1e-6:
            new_angle = np.pi / 2 if new_normal[0] > 0 else -np.pi / 2

        new_window_dict['center'] = new_center
        new_window_dict['normal'] = new_normal
        new_window_dict['size'] = new_size
        new_window_dict['angles'] = [0, 0, new_angle]

        window_normalized_dict = {}
        window_normalized_dict['ID'] = new_window_dict['ID']
        # window_normalized_dict['class'] = window.one_hot_label(all_class_labels)
        window_normalized_dict['class'] = 'window'
        window_normalized_dict['center'] = (new_center / scene_bbox_size).tolist()
        window_normalized_dict['angles'] = [np.cos(new_angle), np.sin(new_angle)]
        window_normalized_dict['size'] = (new_window_dict['size'] / scene_bbox_size).tolist()

        obj_dict_lst.append(new_window_dict)
        obj_normalized_dict_lst.append(window_normalized_dict)

    for obj in object_lst:
        # mesh = obj.raw_model_transformed()
        # obj_size = mesh.bounding_box_oriented.extents
        # print(f'obj size: {obj_size}')
        obj_center = (obj.centroid() - scene_bbox_center)
        obj_size = obj.size * 2
        original_angle = obj.z_angle

        old_obj_dict = {}
        old_obj_dict['ID'] = obj.model_uid
        old_obj_dict['class'] = obj.label
        old_obj_dict['center'] = obj_center.tolist()
        old_obj_dict['angles'] = [0, original_angle, 0]
        old_obj_dict['size'] = obj_size

        new_center = R @ (obj.centroid() - scene_bbox_center)
        rotation_matrix = euler_angle_to_matrix([0, original_angle, 0])
        # forward direction is positive z-axis in 3D-Front
        recovered_normal = rotation_matrix.dot(np.array([0, 0, 1]))
        new_normal = R @ recovered_normal
        new_size = [obj_size[0], obj_size[2], obj_size[1]]
        # now we define the forward direction is negative y-axis.
        cos_angle = new_normal.dot(np.array([0, -1, 0]))
        new_angle = np.arccos(cos_angle)
        if abs(cos_angle) < 1e-6:
            new_angle = np.pi / 2 if new_normal[0] > 0 else -np.pi / 2

        new_obj_dict = {}
        new_obj_dict['ID'] = obj.model_uid
        new_obj_dict['class'] = obj.label
        new_obj_dict['center'] = (new_center).tolist()
        new_obj_dict['angles'] = [0, 0, new_angle]
        new_obj_dict['size'] = new_size

        obj_noromalized_dict = {}
        obj_noromalized_dict['ID'] = obj.model_uid
        # obj_noromalized_dict['class'] = obj.one_hot_label(all_class_labels)
        obj_noromalized_dict['class'] = obj.label
        obj_noromalized_dict['center'] = (new_center / scene_bbox_size).tolist()
        obj_noromalized_dict['angles'] = [np.cos(new_angle), np.sin(new_angle)]
        obj_noromalized_dict['size'] = (new_size / scene_bbox_size).tolist()

        obj_dict_lst.append(new_obj_dict)
        obj_normalized_dict_lst.append(obj_noromalized_dict)

    return {'objects': obj_dict_lst}, {'objects': obj_normalized_dict_lst}


def process_dataset(dataset: ThreedFront, args: argparse.Namespace, split: str = 'train'):
    lack_window_room_num = 0
    lack_door_room_num = 0

    # rotate the coordinate system to x-right, y-forward, z-up
    rotation_to_nerf = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

    all_class_labels_lst = dataset.class_labels
    all_class_order_dict = dataset.class_order
    all_scene_type_lst = dataset.room_types
    max_wall_num = dataset.max_wall_length
    max_furniture_num = dataset.max_furniture_length

    text_encoder = FrozenCLIPEmbedder(device='cuda')

    room_type_str = 'bedroom'
    if args.dataset_filtering == "threed_front_bedroom":
        room_type_str = 'bedroom'
    elif args.dataset_filtering == "threed_front_livingroom":
        room_type_str = 'livingroom'
    elif args.dataset_filtering == "threed_front_diningroom":
        room_type_str = 'diningroom'
    elif args.dataset_filtering == "threed_front_library":
        room_type_str = 'library'
    else:
        raise NotImplementedError

    base_folder_path = os.path.join(args.output_directory, split, room_type_str)
    quad_walls_folder_path = os.path.join(base_folder_path, "quad_walls")
    cam_pose_folder_path = os.path.join(base_folder_path, "cam_pose")
    objects_bbox_folder_path = os.path.join(base_folder_path, "bbox_3d")
    text_desc_folder_path = os.path.join(base_folder_path, "text_desc")
    text_desc_emb_folder_path = os.path.join(base_folder_path, "text_desc_emb")
    img_folder_path = os.path.join(base_folder_path, "img")
    mesh_folder_path = os.path.join(base_folder_path, "mesh")

    os.makedirs(base_folder_path, exist_ok=True)
    os.makedirs(quad_walls_folder_path, exist_ok=True)
    os.makedirs(cam_pose_folder_path, exist_ok=True)
    os.makedirs(objects_bbox_folder_path, exist_ok=True)
    os.makedirs(text_desc_folder_path, exist_ok=True)
    os.makedirs(text_desc_emb_folder_path, exist_ok=True)
    os.makedirs(img_folder_path, exist_ok=True)
    os.makedirs(mesh_folder_path, exist_ok=True)

    room_layout_min_size = np.array([10000000] * 3)
    room_layout_max_size = np.array([-10000000] * 3)
    room_layout_size_lst = []
    for i, ss in tqdm(enumerate(dataset)):
        # if ss.uid != 'ff92de73-ae8f-4ea1-a936-40cbf888f6b4_SecondBedroom-1116':
        #     continue
        # Create a separate folder for each room
        room_name = ss.uid
        json_fname = room_name + '.json'
        img_fname = room_name + '.png'
        mesh_fname = room_name + '.ply'
        txt_fname = room_name + '.txt'
        npy_fname = room_name + '.npy'

        save_quad_wall_filepath = os.path.join(quad_walls_folder_path, json_fname)
        save_object_filepath = os.path.join(objects_bbox_folder_path, json_fname)
        save_cam_pos_filepath = os.path.join(cam_pose_folder_path, txt_fname)
        # save_room_mask_filepath = os.path.join(img_folder_path, "room_mask.png")
        save_rendered_image_filepath = os.path.join(img_folder_path, img_fname)
        save_text_desc_filepath = os.path.join(text_desc_folder_path, txt_fname)
        save_text_desc_emb_filepath = os.path.join(text_desc_emb_folder_path, npy_fname)
        save_debug_scene_bbox_filepath = os.path.join(mesh_folder_path, room_name + '_bbox.ply')
        save_debug_scene_mesh_filepath = os.path.join(mesh_folder_path, room_name + '_mesh.ply')

        # calculate the min/max size of the room layout
        scene_bbox_size = np.array([ss.scene_bbox_size[0], ss.scene_bbox_size[2], ss.scene_bbox_size[1]])
        room_layout_min_size = np.minimum(scene_bbox_size, room_layout_min_size)
        room_layout_max_size = np.maximum(scene_bbox_size, room_layout_max_size)
        room_layout_size_lst.append(scene_bbox_size)

        # save quad walls bbox
        quad_wall_dict, quad_wall_n_dict = parse_room_layout(room=ss,
                                                             all_class_labels=all_class_labels_lst,
                                                             R=rotation_to_nerf)
        with open(save_quad_wall_filepath, "w") as f:
            json.dump(quad_wall_n_dict, f, indent=4)

        # save furtniture bbox
        obj_bbox_dict, obj_bbox_n_dict = parse_bbox_in_room(room=ss,
                                                            all_class_labels=all_class_labels_lst,
                                                            R=rotation_to_nerf)
        with open(save_object_filepath, "w") as f:
            json.dump(obj_bbox_n_dict, f, indent=4)

        # save room centroid
        camera_position = np.array([ss.scene_bbox_centroid[0], -ss.scene_bbox_centroid[2], ss.scene_bbox_centroid[1]])
        np.savetxt(save_cam_pos_filepath, camera_position, fmt='%.6f')

        # generate scene description
        scene_desc_text, scene_desc_emb = get_scene_description(
            room_type=room_type_str,
            wall_dict=copy.deepcopy(quad_wall_dict),
            object_dict=copy.deepcopy(obj_bbox_dict),
            glove_model=text_encoder,
            eval= split == 'test',
        )
        print(f'room {room_name} scene_desc_text: {scene_desc_text}')
        # write text description
        with open(save_text_desc_filepath, 'w') as f:
            f.write(scene_desc_text)
        # write text embedding
        np.save(save_text_desc_emb_filepath, scene_desc_emb)

        # debug
        debug_bbox_lst = quad_wall_dict['walls'] + obj_bbox_dict['objects']
        # debug_bbox_lst = quad_wall_n_dict['walls'] + obj_bbox_n_dict['objects']
        vis_pano_img = np.zeros((512, 1024, 3), dtype=np.uint8)
        vis_pano_img = vis_floor_ceiling_simple(vis_pano_img, color_to_labels=TDFRONT_COLOR_TO_ADEK_LABEL)
        vis_pano_img = vis_objs3d(image=vis_pano_img,
                                  v_bbox3d=debug_bbox_lst,
                                  color_to_labels=TDFRONT_COLOR_TO_ADEK_LABEL,
                                  camera_position=np.array([0, 0, 0]),
                                  b_show_axes=False,
                                  b_show_bbox3d=False,
                                  b_show_centroid=False,
                                  b_show_info=False,
                                  b_show_polygen=True)
        Image.fromarray(vis_pano_img).save(save_rendered_image_filepath)

        new_room_layout_vertices = (rotation_to_nerf @ (ss.wall_meshes.vertices - ss.scene_bbox_centroid).T).T
        new_room_layout_normals = (rotation_to_nerf @ (ss.wall_meshes.vertex_normals.T)).T
        trans_room_layout_mesh = trimesh.Trimesh(vertices=new_room_layout_vertices,
                                                 faces=ss.wall_meshes.faces,
                                                 vertex_normals=new_room_layout_normals)

        scene_bbox = vis_scene_mesh(room_layout_mesh=None,
                                    color_to_labels=TDFRONT_COLOR_TO_ADEK_LABEL,
                                    obj_bbox_lst=debug_bbox_lst)
        # scene_bbox.export(save_debug_scene_bbox_filepath)
        o3d_scene_bbox = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(scene_bbox.vertices),
                                                triangles=o3d.utility.Vector3iVector(scene_bbox.faces))
        o3d_scene_bbox.vertex_normals = o3d.utility.Vector3dVector(scene_bbox.vertex_normals)
        o3d_scene_bbox.compute_triangle_normals()
        o3d_scene_bbox.vertex_colors = o3d.utility.Vector3dVector(scene_bbox.visual.vertex_colors[:, :3] / 255.0)
        o3d.io.write_triangle_mesh(save_debug_scene_bbox_filepath, o3d_scene_bbox)

        # debug furniture mesh
        obj_mesh_lst = []
        for object in ss.bboxes:
            obj_mesh = object.raw_model_transformed()
            new_object_vertices = (rotation_to_nerf @ (obj_mesh.vertices - ss.scene_bbox_centroid).T).T
            new_object_normals = (rotation_to_nerf @ (obj_mesh.vertex_normals.T)).T
            new_object_face_normals = (rotation_to_nerf @ (obj_mesh.face_normals.T)).T
            tr_obj_mesh = trimesh.Trimesh(vertices=new_object_vertices,
                                          faces=obj_mesh.faces,
                                          vertex_normals=new_object_normals,
                                          face_normals=new_object_face_normals)
            labels_lst = list(TDFRONT_COLOR_TO_ADEK_LABEL.values())
            colors_lst = list(TDFRONT_COLOR_TO_ADEK_LABEL.keys())
            color = colors_lst[labels_lst.index(object.label)]
            tr_obj_mesh.visual.face_colors = color

            obj_mesh_lst.append(tr_obj_mesh)
        obj_mesh = trimesh.util.concatenate(obj_mesh_lst) if len(obj_mesh_lst) > 0 else None
        if obj_mesh is not None:
            # debug_bbox_lst = quad_wall_dict['walls'] + [
            #     obj for obj in obj_bbox_dict['objects'] if obj['class'] in ['door', 'window']
            # ]
            # quad_wall_bbox = vis_scene_mesh(room_layout_mesh=None,
            #                                 color_to_labels=TDFRONT_COLOR_TO_ADEK_LABEL,
            #                                 obj_bbox_lst=debug_bbox_lst)
            # obj_mesh = trimesh.util.concatenate([obj_mesh, quad_wall_bbox])
            # obj_mesh.export(save_debug_scene_mesh_filepath)
            o3d_mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(obj_mesh.vertices),
                                                 triangles=o3d.utility.Vector3iVector(obj_mesh.faces))
            o3d_mesh.vertex_normals = o3d.utility.Vector3dVector(obj_mesh.vertex_normals)
            o3d_mesh.compute_triangle_normals()
            o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(obj_mesh.visual.vertex_colors[:, :3] / 255.0)
            o3d.io.write_triangle_mesh(save_debug_scene_mesh_filepath, o3d_mesh)

        # save mesh in 3D-Front coordinate system
        # # save room floor plan as a mesh
        # floor_plan_vertices, floor_plan_faces = ss.floor_plan
        # floor_mesh = trimesh.Trimesh(vertices=floor_plan_vertices, faces=floor_plan_faces)
        # floor_mesh.export(os.path.join(room_directory, "floor.ply"))

        # # save room walls as a mesh
        # path_to_walls_mesh = "{}/walls.ply".format(room_directory)
        # ss.wall_meshes.export(path_to_walls_mesh)

        # path_to_quad_wall_mesh = "{}/quad_wall.ply".format(room_directory)
        # quad_wall_mesh = trimesh.util.concatenate(ss.quad_wall_meshes)
        # quad_wall_mesh.export(path_to_quad_wall_mesh)

        # save door and windows in the room
        if ss.doors_mesh is not None:
            doors_mesh_lst = []
            # path_to_door_mesh = "{}/door.ply".format(room_directory)
            # for idx, mesh in enumerate(ss.doors_mesh):
            #     doors_mesh_lst.append(mesh)
            # doors_mesh = trimesh.util.concatenate(doors_mesh_lst)
            # doors_mesh.export(path_to_door_mesh)
        else:
            # print(f"No door in this room {ss.uid}")
            lack_door_room_num += 1

        if ss.windows_mesh is not None:
            windows_mesh_lst = []
            # path_to_window_mesh = "{}/window.ply".format(room_directory)
            # for idx, mesh in enumerate(ss.windows_mesh):
            #     windows_mesh_lst.append(mesh)
            # windows_mesh = trimesh.util.concatenate(windows_mesh_lst)
            # windows_mesh.export(path_to_window_mesh)
        else:
            # print(f"No window in this room {ss.uid}")
            lack_window_room_num += 1

    # Compute the bounds for the translations, sizes and angles in the dataset.
    # This will then be used to properly align rooms.
    tr_bounds = dataset.bounds["translations"]
    si_bounds = dataset.bounds["sizes"]
    an_bounds = dataset.bounds["angles"]

    dataset_stats = {
        "bounds_translations": tr_bounds[0].tolist() + tr_bounds[1].tolist(),
        "bounds_sizes": si_bounds[0].tolist() + si_bounds[1].tolist(),
        "bounds_angles": an_bounds[0].tolist() + an_bounds[1].tolist(),
        "class_labels": dataset.class_labels,
        "object_types": dataset.object_types,
        "class_frequencies": dataset.class_frequencies,
        "class_order": dataset.class_order,
        "count_furniture": dataset.count_furniture,
        "room_layout_size": room_layout_min_size.tolist() + room_layout_max_size.tolist(),
        "room_layout_size_mean": np.mean(room_layout_size_lst, axis=0).tolist(),
    }
    print(dataset_stats)

    dataset_stat_filepath = os.path.join(base_folder_path, "dataset_stats.json")
    with open(dataset_stat_filepath, "w") as f:
        json.dump(dataset_stats, f, indent=4)

    return lack_door_room_num, lack_window_room_num


def main(argv):
    parser = argparse.ArgumentParser(description="Prepare the 3D-FRONT scenes to train our model")
    parser.add_argument("--output_directory",
                        default="/mnt/nas_3dv/hdd1/datasets/3D_FRONT_FUTURE/",
                        help="Path to output directory")
    parser.add_argument("--path_to_3d_front_dataset_directory",
                        default="/data/dataset/3D_FRONT_FUTURE/3D_Front/3D-FRONT/",
                        help="Path to the 3D-FRONT dataset")
    parser.add_argument("--path_to_3d_future_dataset_directory",
                        default="/data/dataset/3D_FRONT_FUTURE/3D_Future/3D-FUTURE-model",
                        help="Path to the 3D-FUTURE dataset")
    parser.add_argument("--path_to_model_info",
                        default="/data/dataset/3D_FRONT_FUTURE/3D_Future/3D-FUTURE-model/model_info.json",
                        help="Path to the 3D-FUTURE model_info.json file")
    parser.add_argument("--path_to_floor_plan_textures",
                        default="/home/hkust/fangchuan/codes/ATISS/demo/",
                        help="Path to floor texture images")
    parser.add_argument("--path_to_invalid_scene_ids",
                        default="../config/invalid_threed_front_rooms.txt",
                        help="Path to invalid scenes")
    parser.add_argument("--path_to_invalid_bbox_jids",
                        default="../config/threed_front_black_list.txt",
                        help="Path to objects that ae blacklisted")
    parser.add_argument("--annotation_file",
                        default="../config/bedroom_threed_front_splits.csv",
                        help="Path to the train/test splits file")
    parser.add_argument("--room_side", type=float, default=3.1, help="The size of the room along a side (default:3.1)")
    parser.add_argument(
        "--dataset_filtering",
        default="threed_front_bedroom",
        choices=["threed_front_bedroom", "threed_front_livingroom", "threed_front_diningroom", "threed_front_library"],
        help="The type of dataset filtering to be used")
    parser.add_argument("--without_lamps", action="store_true", help="If set ignore lamps when rendering the room")
    parser.add_argument("--up_vector",
                        type=lambda x: tuple(map(float, x.split(","))),
                        default="0,0,-1",
                        help="Up vector of the scene")
    parser.add_argument("--background",
                        type=lambda x: list(map(float, x.split(","))),
                        default="0,0,0,1",
                        help="Set the background of the scene")
    parser.add_argument("--camera_target",
                        type=lambda x: tuple(map(float, x.split(","))),
                        default="0,0,0",
                        help="Set the target for the camera")
    parser.add_argument("--camera_position",
                        type=lambda x: tuple(map(float, x.split(","))),
                        default="0,4,0",
                        help="Camer position in the scene")
    parser.add_argument("--window_size",
                        type=lambda x: tuple(map(int, x.split(","))),
                        default="256,256",
                        help="Define the size of the scene and the window")

    args = parser.parse_args(argv)
    logging.getLogger("trimesh").setLevel(logging.ERROR)

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    config = {
        "filter_fn": args.dataset_filtering,
        "min_n_boxes": -1,
        "max_n_boxes": -1,
        "path_to_invalid_scene_ids": args.path_to_invalid_scene_ids,
        "path_to_invalid_bbox_jids": args.path_to_invalid_bbox_jids,
        "annotation_file": args.annotation_file
    }

    train_dataset = ThreedFront.from_dataset_directory(dataset_directory=args.path_to_3d_front_dataset_directory,
                                                       path_to_model_info=args.path_to_model_info,
                                                       path_to_models=args.path_to_3d_future_dataset_directory,
                                                       filter_fn=filter_function(config, TRAIN_SCENE,
                                                                                 args.without_lamps))
    test_dataset = ThreedFront.from_dataset_directory(dataset_directory=args.path_to_3d_front_dataset_directory,
                                                      path_to_model_info=args.path_to_model_info,
                                                      path_to_models=args.path_to_3d_future_dataset_directory,
                                                      filter_fn=filter_function(config, TEST_SCENE, args.without_lamps))

    print("Loading train/val dataset with {} rooms".format(len(train_dataset)))
    print("Loading test dataset with {} rooms".format(len(test_dataset)))

    lack_door_n1, lack_window_n1 = process_dataset(train_dataset, args, split='train')
    lack_door_n2, lack_window_n2 = process_dataset(test_dataset, args, split='test')
    lack_door_room_num = lack_door_n1 + lack_door_n2
    lack_window_room_num = lack_window_n1 + lack_window_n2

    print(f"lack door room num: {lack_door_room_num}")
    print(f"lack window room num: {lack_window_room_num}")


if __name__ == "__main__":
    main(sys.argv[1:])
