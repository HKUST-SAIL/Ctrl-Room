import os
import sys

sys.path.append(".")
sys.path.append("..")  # Adds higher directory to python modules path.
import argparse

import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib as matlib

from shapely.geometry import Polygon
from descartes.patch import PolygonPatch
from tqdm import tqdm
from PIL import Image
import open3d as o3d
import trimesh
import json

from dataset.metadata import (ST3D_BEDROOM_FURNITURE, ST3D_LIVINGROOM_FURNITURE, ST3D_DININGROOM_FURNITURE, ST3D_KITCHEN_FURNITURE,
                              ST3D_BEDROOM_QUAD_WALL_MAX_LEN, ST3D_LIVINGROOM_QUAD_WALL_MAX_LEN, COLOR_TO_ADEK_LABEL)
from dataset.threed_front.metadata import (THREED_FRONT_BEDROOM_FURNITURE, THREED_FRONT_DININGROOM_FURNITURE,
                                           THREED_FRONT_LIVINGROOM_FURNITURE, THREED_FRONT_BEDROOM_MAX_WALL_NUM, THREED_FRONT_LIVINGROOM_MAX_WALL_NUM)
from dataset.threed_front.threed_future_dataset import ThreedFutureDataset
from dataset.st3d_dataset import ST3DDataset, np_coor2xy, np_coor2xy, ROOM_TYPE_DICT
from preprocess.prepare_st3d_dataset import vis_scene_mesh
from utils.equirect_projection import vis_objs3d, vis_floor_ceiling_simple
from utils.utils import euler_angle_to_matrix, reconstrcut_floor_ceiling_from_quad_walls


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir',
                        default='/mnt/nas_3dv/hdd1/datasets/Structured3d/preprocessed/new_sem_layout/test/bedroom/')
    parser.add_argument('--samples_filepath',
                        default='../sample_results/openai-2023-08-08-14-07-51-332048/samples_10x23x33.npz')
    parser.add_argument('--path_to_pickled_3d_futute_models',
                        default='/mnt/nas_3dv/hdd1/datasets/3D_FRONT_FUTURE/threed_future_model_bedroom.pkl',
                        type=str,
                        help='3D_FURTURE models')
    parser.add_argument('--dataset_type', default='st3d', type=str, help='dataset type, [3d_front, st3d]')
    parser.add_argument('--room_type', default='bedroom', type=str, help='generated room type')
    parser.add_argument('--vis_layout_mesh', action='store_true', help='whether to visualize layout mesh')
    parser.add_argument('--vis_layout_wireframe', action='store_true', help='whether to visualize wireframe of layout')
    parser.add_argument('--out_dir', default='sample_dataset_visualization')
    return parser.parse_args()


def visualize_a_data(img, bound_y_lst, corner_y_lst):
    img = (img.numpy().transpose([1, 2, 0]) * 255).astype(np.uint8)
    img_H, img_W = img.shape[:2]
    bound_y_lst = bound_y_lst.numpy()
    # scale to image pixel coordinates
    bound_y_lst = ((bound_y_lst / np.pi + 0.5) * img_H).round().astype(int)
    corner_y_lst = corner_y_lst.numpy()

    git_corner_img = np.zeros((30, 1024, 3), np.uint8)
    git_corner_img[:] = corner_y_lst[0][None, :, None] * 255
    padding_img = np.zeros((3, 1024, 3), np.uint8) + 255

    img_with_boundary = (img.copy() * 0.5).astype(np.uint8)
    y1 = np.round(bound_y_lst[0]).astype(int)
    y2 = np.round(bound_y_lst[1]).astype(int)
    y1 = np.vstack([np.arange(1024), y1]).T.reshape(-1, 1, 2)
    y2 = np.vstack([np.arange(1024), y2]).T.reshape(-1, 1, 2)
    img_with_boundary[bound_y_lst[0], np.arange(len(bound_y_lst[0])), 1] = 255
    img_with_boundary[bound_y_lst[1], np.arange(len(bound_y_lst[1])), 1] = 255

    return np.concatenate([git_corner_img, padding_img, img_with_boundary], 0)


def visualize_synth_data_4_1024(bound_y_lst, corner_y_lst, obj_bbox_lst, cam_position):
    img = np.zeros((512, 1024, 3), np.uint8)
    img_H, img_W = img.shape[:2]
    # scale to image pixel coordinates
    bound_y_lst = ((bound_y_lst / np.pi + 0.5) * img_H).round().astype(int)

    git_corner_img = np.zeros((30, 1024, 3), np.uint8)
    git_corner_img[:] = corner_y_lst[None, :, None] * 255
    padding_img = np.zeros((3, 1024, 3), np.uint8) + 255

    img_with_boundary = (img.copy() * 0.5).astype(np.uint8)
    # draw boundary lines green
    img_with_boundary[bound_y_lst[0], np.arange(len(bound_y_lst[0])), 1] = 255
    img_with_boundary[bound_y_lst[1], np.arange(len(bound_y_lst[1])), 1] = 255

    ret_img = np.concatenate([git_corner_img, padding_img, img_with_boundary], 0)
    ret_img = vis_objs3d(ret_img,
                         v_bbox3d=obj_bbox_lst,
                         camera_position=cam_position,
                         color_to_labels=None,
                         b_show_axes=False,
                         b_show_centroid=False,
                         b_show_bbox3d=True,
                         b_show_info=True)
    return ret_img


# save layout mesh
def save_layout_mesh(save_filepath: str, points: np.array, faces: np.array):
    ply_header = '\n'.join([
        'ply',
        'format ascii 1.0',
        f'element vertex {len(points):d}',
        'property float x',
        'property float y',
        'property float z',
        'property uchar red',
        'property uchar green',
        'property uchar blue',
        f'element face {len(faces):d}',
        'property list uchar int vertex_indices',
        'end_header',
    ])
    with open(save_filepath, 'w') as f:
        f.write(ply_header)
        f.write('\n')
        for x, y, z, r, g, b in points:
            f.write(f'{x:.2f} {y:.2f} {z:.2f} {r:.0f} {g:.0f} {b:.0f}\n')
        for i, j, k in faces:
            f.write(f'3 {i:d} {j:d} {k:d}\n')


def recover_quad_wall_layout_mesh(dataset_type: str,
                                  room_type: str,
                                  quad_wall_lst: np.ndarray,
                                  object_bbox_lst: np.ndarray,
                                  room_layout_bbox_size: np.array = np.array([1.0, 1.0, 1.0])):

    if room_type == 'bedroom':
        class_labels_lst = (ST3D_BEDROOM_FURNITURE) if dataset_type == 'st3d' else THREED_FRONT_BEDROOM_FURNITURE
    elif room_type == 'livingroom':
        class_labels_lst = (ST3D_LIVINGROOM_FURNITURE) if dataset_type == 'st3d' else THREED_FRONT_LIVINGROOM_FURNITURE
    elif room_type == 'diningroom':
        class_labels_lst = (ST3D_DININGROOM_FURNITURE) if dataset_type == 'st3d' else THREED_FRONT_DININGROOM_FURNITURE
    elif room_type == 'kitchen':
        class_labels_lst = (ST3D_KITCHEN_FURNITURE) if dataset_type == 'st3d' else THREED_FRONT_KITCHEN_FURNITURE
    else:
        raise NotImplementedError

    print(f' room_type: {room_type}, class_labels_lst: {len(class_labels_lst)}')
    class_idx = 0
    centroid_idx = len(class_labels_lst)
    size_idx = 3 + centroid_idx
    angle_idx = 3 + size_idx

    # recover quad wall bbox of room layout
    quad_wall_dict_list = []
    for i in range(len(quad_wall_lst)):
        # print(f'quad wall: {quad_wall_lst[i]}')
        quad_wall_dict = {}
        # recover class label
        class_label_prob = quad_wall_lst[i][:centroid_idx]
        # print(f'class_label_prob: {class_label_prob}')
        class_label_prob = np.where(class_label_prob > 0.5, class_label_prob, 0)
        if np.all(class_label_prob == 0):
            print(f'wall {i} has no class label')
            continue
        class_label = class_labels_lst[class_label_prob.argmax()]
        if class_label == 'empty':
            continue
        quad_wall_dict['class'] = class_label
        wall_center = quad_wall_lst[i][centroid_idx:centroid_idx + 3] * room_layout_bbox_size
        quad_wall_dict['center'] = wall_center.tolist()
        wall_size = quad_wall_lst[i][size_idx:size_idx + 3]
        # wall_normal_angle = quad_wall_lst[i][angle_idx:angle_idx + 2]

        # angle_0 = np.arccos(wall_normal_angle[0])
        # angle_1 = np.arcsin(wall_normal_angle[1])
        # angle = angle_1 if abs(wall_normal_angle[0]) < 5e-3 else angle_0
        angle = quad_wall_lst[i][angle_idx]

        quad_wall_dict['angles'] = [0, 0, float(angle)]
        # The direction of all camera is always along the negative y-axis.
        rotation_matrix = euler_angle_to_matrix(quad_wall_dict['angles'])
        wall_normal = rotation_matrix.dot(np.array([0, -1, 0]))
        # print(f'wall_normal: {wall_normal}')
        wall_size = np.array([wall_size[0], 0.01, wall_size[2]])
        wall_size = wall_size * room_layout_bbox_size
        wall_corners = np.array([
            [-wall_size[0] / 2, 0, -wall_size[2] / 2],  # left-bottom
            [-wall_size[0] / 2, 0, wall_size[2] / 2],  # left-top
            [wall_size[0] / 2, 0, wall_size[2] / 2],  # right-top
            [wall_size[0] / 2, 0, -wall_size[2] / 2]
        ])  # right-bottom
        wall_corners = wall_corners.dot(rotation_matrix.T)
        wall_corners = wall_corners + wall_center
        quad_wall_dict['size'] = wall_size.tolist()
        quad_wall_dict['corners'] = wall_corners.tolist()
        quad_wall_dict['normal'] = wall_normal.tolist()
        test_width = np.linalg.norm(wall_corners[0] - wall_corners[3])
        test_height = np.linalg.norm(wall_corners[0] - wall_corners[1])
        # print(f'with: {wall_size[0]} height: {wall_size[2]}')
        # print(f'test_width: {test_width} test_height: {test_height}')
        # print(f' wall {class_label} centroid: {wall_center} size: {wall_size} noraml: {wall_normal}')
        quad_wall_dict_list.append(quad_wall_dict)
    print(f'quad walls num: {len(quad_wall_dict_list)}')

    # recover object bbox
    obj_bbox_dict_list = []
    door_cnt = 0
    table_cnt = 0
    for i in range(len(object_bbox_lst)):
        # print(f'predict object bbox feature: {object_bbox_lst[i]}')
        obj_bbox_dict = {}

        # recover class label
        class_label_prob = object_bbox_lst[i][:centroid_idx]
        # print(f'class_label_prob: {class_label_prob}')
        class_label_prob = np.where(class_label_prob > 0.5, class_label_prob, 0)
        # if len(class_label_prob) == 0:
        #     print(f'object {i} has no class label')
        if np.all(class_label_prob == 0):
            print(f'object {i} has no class label')
            continue
        class_label = class_labels_lst[class_label_prob.argmax()]
        if class_label == 'empty':
            continue
        # if class_label == 'door':
        #     door_cnt += 1
        #     if door_cnt < 3:
        #         continue
        # if class_label == 'shelves':
        #         continue
        obj_bbox_dict['class'] = class_label

        # recover centroid
        centroid = object_bbox_lst[i][centroid_idx:size_idx]
        centroid = centroid * room_layout_bbox_size
        obj_bbox_dict['center'] = centroid.tolist()
        # recover size
        size = object_bbox_lst[i][size_idx:angle_idx]
        size = size * room_layout_bbox_size
        obj_bbox_dict['size'] = size.tolist()
        # recover angle
        # cs_angle_value = object_bbox_lst[i][angle_idx:]
        # angle_0 = np.arccos(cs_angle_value[0])
        # angle_1 = np.arcsin(cs_angle_value[1])
        # angle = angle_1 if abs(cs_angle_value[0]) < 5e-3 else angle_0
        angle = object_bbox_lst[i][angle_idx]
        obj_bbox_dict['angles'] = [0, 0, float(angle)]
        # print(f' object {class_label} centroid: {centroid} size: {size} angle: {angle_0}')
        obj_bbox_dict_list.append(obj_bbox_dict)
    print(f'object num: {len(obj_bbox_dict_list)}')

    return quad_wall_dict_list, obj_bbox_dict_list


def get_textured_objects(bbox_params_dict, objects_dataset: ThreedFutureDataset, color_to_labels: dict = None):
    # For each one of the boxes replace them with an object
    renderables = []
    trimesh_meshes = []

    # rotate the coordinate system to x-right, y-forward, z-up
    rotation_to_nerf = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

    for i in range(len(bbox_params_dict)):
        query_label = bbox_params_dict[i]['class']
        if query_label in ['door', 'window', 'wall']:
            continue

        query_size = bbox_params_dict[i]['size']
        query_size = np.array([query_size[0] / 2, query_size[2] / 2, query_size[1] / 2])
        query_translation = np.array(bbox_params_dict[i]['center'])
        query_rotation = bbox_params_dict[i]['angles']
        # 3D-FURTURE object is in x-right, y-up, z-forward coordinate system
        query_rotation = euler_angle_to_matrix(query_rotation)
        # print(f'query_label: {query_label} query_size: {query_size}')
        furniture = objects_dataset.get_closest_furniture_to_box(query_label, query_size)
        # print(f'retrieved furniture: {furniture.size}')

        # Create a trimesh object for the same mesh in order to save
        # everything as a single scene
        tr_mesh = trimesh.load(furniture.raw_model_path, force="mesh")
        centroid = tr_mesh.bounding_box.centroid
        tr_mesh.vertices *= furniture.scale
        tr_mesh.vertices -= centroid
        # rotate the coordinate system to x-right, y-forward, z-up
        new_tr_mesh_vertices = rotation_to_nerf.dot(tr_mesh.vertices.T).T
        new_tr_mesh_vertex_normals = rotation_to_nerf.dot(tr_mesh.vertex_normals.T).T
        new_tr_mesh_face_normals = rotation_to_nerf.dot(tr_mesh.face_normals.T).T
        tr_mesh = trimesh.Trimesh(vertices=new_tr_mesh_vertices,
                                  faces=tr_mesh.faces,
                                  vertex_normals=new_tr_mesh_vertex_normals,
                                  face_normals=new_tr_mesh_face_normals)

        tr_mesh.vertices[...] = tr_mesh.vertices.dot(query_rotation) + query_translation
        if color_to_labels is not None:
            labels_lst = list(color_to_labels.values())
            colors_lst = list(color_to_labels.keys())
            color = colors_lst[labels_lst.index(query_label)]
            # tr_mesh.visual.face_colors = color
            tr_mesh.visual.vertex_colors = color
        else:
            tr_mesh.visual.material.image = Image.open(furniture.texture_image_path)

        trimesh_meshes.append(tr_mesh)

    return trimesh_meshes


if __name__ == "__main__":

    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print('args:')
    for key, val in vars(args).items():
        print('    {:16} {}'.format(key, val))

    dataset = ST3DDataset(root_dir=args.root_dir)
    # Showing some information about dataset
    print('len(dataset): {}'.format(len(dataset)))

    # Build the dataset of 3D models
    threed_furture_dataset = ThreedFutureDataset.from_pickled_dataset(args.path_to_pickled_3d_futute_models)
    print("Loaded {} 3D-FUTURE models".format(len(threed_furture_dataset)))

    b_vis_mesh_from_corners = False
    b_vis_mesh_from_diffusion = True

    # load sample results: BxNxChannel
    sample_result_lst = np.load(args.samples_filepath)

    for idx in range(len(sample_result_lst['arr_0'])):
        scene_sample_result = sample_result_lst['arr_0'][idx]
        scene_sample_label = args.room_type
        print(f'scene_sample_label: {scene_sample_label}')

        if args.dataset_type == 'st3d':
            if args.room_type == 'bedroom':
                room_layout_size = np.array([3.64073229, 3.73553261, 2.81591231])  # bedroom
                wall_max_num = ST3D_BEDROOM_QUAD_WALL_MAX_LEN
            elif args.room_type == 'livingroom':
                room_layout_size = np.array([7.239328956952291, 7.6231320936720675, 2.857928654068745])  # livingroom
                wall_max_num = ST3D_LIVINGROOM_QUAD_WALL_MAX_LEN
            elif args.room_type == 'kitchen':
                room_layout_size = np.array([5.190448841484693, 5.507005307872562, 2.6443791106290626])
                wall_max_num = ST3D_BEDROOM_QUAD_WALL_MAX_LEN
        
        if args.dataset_type == '3d_front':
            if args.room_type == 'bedroom':
                room_layout_size = np.array([3.5491398691635867,3.8409623633141603, 2.651076370213072])  # bedroom
                wall_max_num = THREED_FRONT_BEDROOM_MAX_WALL_NUM
            elif args.room_type == 'livingroom':
                room_layout_size = np.array([5.578468844343161, 6.169652242321863, 2.635967136258661])  # livingroom
                wall_max_num = THREED_FRONT_LIVINGROOM_MAX_WALL_NUM
            elif args.room_type == 'diningroom':
                room_layout_size = np.array([5.190448841484693, 5.507005307872562, 2.6443791106290626]) # diningroom
                wall_max_num = THREED_FRONT_LIVINGROOM_MAX_WALL_NUM

        # quad walls
        quad_wall_lst = scene_sample_result[:wall_max_num, :]
        # objects
        obj_bbox_lst = scene_sample_result[wall_max_num:, :]

        if b_vis_mesh_from_corners:
            layout_ply_points, layout_ply_faces, layout_corner_lst, cam_pos_lst = dataset.get_gt_layout_mesh(idx)
        elif b_vis_mesh_from_diffusion:
            # layout_ply_points, layout_ply_faces, layout_corner_lst, cam_pos_lst, room_layout_mesh, obj_bbox_dict_lst = dataset.get_predicted_layout_mesh(
            #     room_type=args.room_type,
            #     bound_ceil_floor_lst=boundary_lst,
            #     wall_prob_lst=wall_prob_lst,
            #     obj_bbox_lst=obj_bbox_lst,
            #     b_force_raw=False)
            wall_dict_lst, obj_bbox_dict_lst = recover_quad_wall_layout_mesh(dataset_type=args.dataset_type,
                                                                             room_type=args.room_type,
                                                                             quad_wall_lst=quad_wall_lst,
                                                                             object_bbox_lst=obj_bbox_lst,)
                                                                            #  room_layout_bbox_size=room_layout_size)

        # save synthetic boundaries as image
        img_fname = f'{scene_sample_label}_{idx}.png'
        # out_img = visualize_synth_data_4_1024(bound_y_lst=boundary_lst,
        #                                       corner_y_lst=wall_prob_lst,
        #                                       obj_bbox_lst=obj_bbox_dict_lst,
        #                                       cam_position=cam_pos_lst)
        out_img = np.zeros((512, 1024, 3), np.uint8)
        cam_position = np.zeros((3,), np.float32)
        reconstrcut_floor_ceiling_from_quad_walls(quad_walls_lst=wall_dict_lst)
        out_img = vis_floor_ceiling_simple(image=out_img, color_to_labels=COLOR_TO_ADEK_LABEL)
        out_img = vis_objs3d(out_img,
                             v_bbox3d=(wall_dict_lst + obj_bbox_dict_lst),
                             camera_position=cam_position,
                             color_to_labels=COLOR_TO_ADEK_LABEL if args.dataset_type == 'st3d' else None,
                             b_show_axes=False,
                             b_show_centroid=False,
                             b_show_bbox3d=False,
                             b_show_info=False,
                             b_show_polygen=True)
        save_img_filepath = os.path.join(args.out_dir, img_fname)
        Image.fromarray(out_img).save(save_img_filepath)

        # scene layout file
        scene_layout_fname = f'{scene_sample_label}_{idx}.json'
        scene_layout_filepath = os.path.join(args.out_dir, scene_layout_fname)
        with open(scene_layout_filepath, 'w') as f:
            json.dump({'walls': wall_dict_lst, 'objects': obj_bbox_dict_lst}, f, indent=4)

        # save synthetic object and room_layout as ply
        scene_bbox_ply_fname = f'{scene_sample_label}_{idx}.ply'
        scene_bbox_ply_filepath = os.path.join(args.out_dir, scene_bbox_ply_fname)
        # save_layout_mesh(save_ply_filepath, layout_ply_points, layout_ply_faces)
        scene_mesh = vis_scene_mesh(room_layout_mesh=None,
                                    obj_bbox_lst=(wall_dict_lst + obj_bbox_dict_lst),
                                    color_to_labels=COLOR_TO_ADEK_LABEL,
                                    room_layout_bbox=None)
        scene_mesh.export(scene_bbox_ply_filepath)

        # search 3D-FUTURE models for objects
        if args.dataset_type == '3d_front':
            objects_mesh_lst = get_textured_objects(obj_bbox_dict_lst, threed_furture_dataset)
            scene_mesh_ply_fname = f'{scene_sample_label}_{idx}_mesh.ply'
            scene_mesh_ply_filepath = os.path.join(args.out_dir, scene_mesh_ply_fname)
            objects_mesh = trimesh.util.concatenate(objects_mesh_lst)
            scene_mesh = vis_scene_mesh(room_layout_mesh=objects_mesh,
                                        obj_bbox_lst=wall_dict_lst,
                                        room_layout_bbox=None)
            scene_mesh.export(scene_mesh_ply_filepath)
