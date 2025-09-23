# recover colored 3D scene from bbox3d.json file
import os
import sys
sys.path.append('.')
sys.path.append('..')
import os.path as osp
import argparse

import numpy as np
import json
from PIL import Image

from utils.vis_utils import save_visualization_and_mesh


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Directory to dataset')
    parser.add_argument('--scene_id', type=str, default='scene_03000_238',
                        help='Target scene id to visualize')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory')
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    scene_id = args.scene_id
    output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # load wall bbox file
    wall_bbox_filepath = osp.join(dataset_dir, 'test/livingroom/quad_walls', f'{scene_id}.json')
    # load object bbox file
    object_bbox_filepath = osp.join(dataset_dir, 'test/livingroom/bbox_3d', f'{scene_id}.json')
    # 2d wall corners label file
    wall_corner_2d_label_filepath = osp.join(dataset_dir, 'test/livingroom/label_cor', f'{scene_id}.txt')
    # pano rgb image file
    rgb_img_filepath = osp.join(dataset_dir, 'test/livingroom/img', f'{scene_id}.png')
    
    
    object_bbox_lst = []
    wall_bbox_lst = []
    with open(object_bbox_filepath) as f:
        bbox_dicts = json.load(f)
        object_bbox_lst = bbox_dicts['objects']
    with open(wall_bbox_filepath) as f:
        bbox_dicts = json.load(f)
        wall_bbox_lst = bbox_dicts['walls']


    sem_layout_img, layout_bbox_trimesh = save_visualization_and_mesh(objects_lst=object_bbox_lst,
                                quad_walls_lst=wall_bbox_lst,
                                source_cor_path=wall_corner_2d_label_filepath,
                                source_img_path=rgb_img_filepath)

    # save semantic layout img
    save_img_filepath = os.path.join(output_dir, f'{scene_id}_sem.png')
    Image.fromarray(sem_layout_img).save(save_img_filepath)
    
    scene_bbox_ply_fname = f'{scene_id}.ply'
    scene_bbox_ply_filepath = os.path.join(output_dir, scene_bbox_ply_fname)
    layout_bbox_trimesh.export(scene_bbox_ply_filepath)
    
