import os
import json
import argparse
import sys

sys.path.append('.')
sys.path.append('..')

import cv2
import numpy as np

from utils.utils import matrix_to_euler_angles
from dataset.metadata import INVALID_SCENES_LST, INVALID_ROOMS_LST, OBJECT_LABEL_IDS
from prepare_st3d_dataset import vis_color_pointcloud

TARGET_ROOM_TYPE_LST = ['living room', 'bedroom', 'dining room', 'kitchen', 'bathroom', 'study']


def visualize_bbox(dataset_folderpath, output_folderpath):

    SCENE_LST = ['scene_%05d' % i for i in range(0, 3500) if ('scene_%05d' % i) not in INVALID_SCENES_LST]

    for scene_id in SCENE_LST:
        scene_img_path = os.path.join(dataset_folderpath, scene_id, "2D_rendering")

        room_type_lst = None
        scene_anno_3d_filepath = os.path.join(dataset_folderpath, scene_id, "annotation_3d.json")
        if not os.path.isfile(scene_anno_3d_filepath):
            INVALID_SCENES_LST.append(scene_id)
            continue
        else:
            scene_anno_3d_dict = json.load(open(scene_anno_3d_filepath, 'r'))
            room_type_lst = scene_anno_3d_dict['semantics']

        for room_id in np.sort(os.listdir(scene_img_path)):
            room_id_str = scene_id + '_' + room_id
            if room_id_str in INVALID_ROOMS_LST:
                continue

            room_path = os.path.join(scene_img_path, room_id, "panorama")
            if not os.path.exists(room_path):
                continue
            room_bbox_3d_path = os.path.join(room_path, 'full', 'bbox_3d.json')
            with open(room_bbox_3d_path, 'r') as file:
                annos = json.load(file)

            id2index = dict()
            for index, object in enumerate(annos):
                id2index[object.get('ID')] = index

            room_type_str = 'undefined'
            if room_type_lst is not None:
                for rt in room_type_lst:
                    if rt['ID'] == int(room_id):
                        room_type_str = rt['type']
                        break
            if room_type_str == 'undefined':
                continue

            print(f'preprocess room : {room_id_str}')

            output_path = os.path.join(output_folderpath, room_type_str)
            output_pcl_path = os.path.join(output_folderpath, room_type_str, 'pointclouds')
            output_label_path = os.path.join(output_folderpath, room_type_str, 'labels')
            output_rgb_path = os.path.join(output_folderpath, room_type_str, 'rgb')
            os.makedirs(output_path, exist_ok=True)
            os.makedirs(output_pcl_path, exist_ok=True)
            os.makedirs(output_label_path, exist_ok=True)
            os.makedirs(output_rgb_path, exist_ok=True)

            rgb_img_path = os.path.join(room_path, 'full', 'rgb_rawlight.png')
            depth_img_path = os.path.join(room_path, 'full', 'depth.png')
            instance_img_path = os.path.join(room_path, 'full', 'instance.png')

            assert os.path.exists(rgb_img_path)
            # assert os.path.exists(depth_img_path)
            if not os.path.exists(depth_img_path):
                print(f'no depth image for {room_id_str}')
                continue
            assert os.path.exists(instance_img_path)

            rgb_img = cv2.imread(rgb_img_path, cv2.IMREAD_UNCHANGED)
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

            instance_img = cv2.imread(instance_img_path, cv2.IMREAD_UNCHANGED)

            # save pointcloud
            pointcloud_filename = room_id_str + '.ply'
            pointcloud_filepath = os.path.join(output_pcl_path, room_id_str + '.ply')
            vis_color_pointcloud(rgb_img_filepath=rgb_img_path,
                                 depth_img_filepath=depth_img_path,
                                 saved_color_pcl_filepath=pointcloud_filepath)

            cam_position = np.loadtxt(os.path.join(room_path, 'camera_xyz.txt'))
            cam_position = cam_position
            # print(f'cam_position: {cam_position}')

            # save object labels
            saved_3dbbox_prior_filepath = os.path.join(output_label_path, room_id_str + '.json')
            obj_bbox_lst = []
            # skip background
            for index in np.unique(instance_img)[:-1]:
                # for each instance in current image
                # we remove some incorrect objeect labels manually
                if index not in id2index.keys():
                    continue
                bbox = annos[id2index[index]]

                if bbox['label'] not in OBJECT_LABEL_IDS.keys():
                    continue

                basis = np.array(bbox['basis'])
                coeffs = np.array(bbox['coeffs'])
                centroid = np.array(bbox['centroid'])

                obj_bbox_dict = {}
                # obj_bbox_dict['rotations'] = basis.tolist()
                # obj_bbox_dict['centroid'] = list(centroid)
                # obj_bbox_dict['dimensions'] = list(coeffs)
                obj_bbox_dict['name'] = bbox['label']

                center = (centroid - cam_position) * 0.001
                obj_bbox_dict['centroid'] = {}
                obj_bbox_dict['centroid']['x'] = float(center[0])
                obj_bbox_dict['centroid']['y'] = float(center[1])
                obj_bbox_dict['centroid']['z'] = float(center[2])
                size = coeffs * 0.001 * 2
                obj_bbox_dict['dimensions'] = {}
                obj_bbox_dict['dimensions']['length'] = float(size[0])
                obj_bbox_dict['dimensions']['width'] = float(size[1])
                obj_bbox_dict['dimensions']['height'] = float(size[2])
                angles = matrix_to_euler_angles(basis).tolist()
                obj_bbox_dict['rotations'] = {}
                obj_bbox_dict['rotations']['x'] = float(angles[0])
                obj_bbox_dict['rotations']['y'] = float(angles[1])
                obj_bbox_dict['rotations']['z'] = float(angles[2])
                # obj_bbox_dict['center'] = list((centroid - cam_position) * 0.001)
                # obj_bbox_dict['size'] = list(coeffs * 0.001 * 2)
                obj_bbox_lst.append(obj_bbox_dict)

            root_node = {}
            root_node['folder'] = 'pointclouds'
            root_node['filename'] = pointcloud_filename
            root_node['path'] = 'pointclouds\\' + pointcloud_filename
            root_node['objects'] = obj_bbox_lst
            with open(saved_3dbbox_prior_filepath, 'w') as fd:
                json.dump(root_node, fd)

            # save rgb panorama
            saved_rgb_img_filepath = os.path.join(output_rgb_path, room_id_str + '.png')
            cv2.imwrite(saved_rgb_img_filepath, rgb_img)
            # if room_type_str != 'undefined':
            #     anno_img = vis_objs3d(image=rgb_img,
            #                           v_bbox3d=obj_bbox_lst,
            #                           camera_position=cam_position,
            #                           b_show_axes=False,
            #                           b_show_centroid=False,
            #                           b_show_bbox3d=True,
            #                           b_show_info=True,
            #                           thickness=2)
            #     output_img_filepath = os.path.join(output_folderpath, room_type_str, room_id_str + '_bbox.png')
            #     print(f'save visualization for object bbox annotation of {room_id_str}')
            #     cv2.imwrite(output_img_filepath, anno_img)
            #     save_obj_bbox_filepath = os.path.join(output_folderpath, room_type_str, room_id_str + '_bbox.json')
            #     obj_bbox_dicts = {}
            #     obj_bbox_dicts['objects'] = obj_bbox_lst
            #     json.dump(obj_bbox_dicts, open(save_obj_bbox_filepath, 'w'), indent=4)


def parse_args():
    parser = argparse.ArgumentParser(description="Structured3D 3D Bounding Box Visualization")
    parser.add_argument("--dataset_path",
                        default="/data/dataset/Structured3D/Structured3D/",
                        help="raw dataset path",
                        metavar="DIR")
    parser.add_argument("--debug_path",
                        default="/data/dataset/Structured3D/preprocessed/annotations",
                        help="debug folder path for object bbox annotations",
                        metavar="DIR")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_folderpath = args.dataset_path
    debug_folderpath = args.debug_path

    if not os.path.exists(dataset_folderpath):
        raise ValueError("Dataset folder does not exist!")

    if not os.path.exists(debug_folderpath):
        os.makedirs(debug_folderpath)

    visualize_bbox(dataset_folderpath, debug_folderpath)


if __name__ == "__main__":
    main()
