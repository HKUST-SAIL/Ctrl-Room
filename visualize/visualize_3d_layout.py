# recover colored 3D scene from bbox3d.json file
import os
import sys
sys.path.append('.')
sys.path.append('..')

import numpy as np
import json
from PIL import Image
from utils.equirect_projection import vis_objs3d
from utils.utils import vis_scene_mesh
from dataset.metadata import COLOR_TO_ADEK_LABEL

# load object bbox file
bbox_filepath = '/mnt/nas_3dv/hdd1/fangchuan/HolisticDiffuScene/scripts/scene_03200_522357.json'
scene_name = os.path.basename(bbox_filepath)[:-5]
object_bbox_lst = []
wall_bbox_lst = []
with open(bbox_filepath) as f:
    bbox_dicts = json.load(f)
    obj_bbox_dict_lst = bbox_dicts['objects']
    wall_dict_lst = bbox_dicts['walls']
    
print(f'wall_dict_lst: {wall_dict_lst}')
print(f'obj_bbox_dict_lst: {obj_bbox_dict_lst}')

# save synthesis results as image
out_img = np.zeros((512, 1024, 3), np.uint8)
cam_position = np.zeros((3,), np.float32)
# post-process walls
out_img = vis_objs3d(out_img,
                        v_bbox3d=(wall_dict_lst + obj_bbox_dict_lst),
                        camera_position=cam_position,
                        color_to_labels=COLOR_TO_ADEK_LABEL,
                        b_show_axes=False,
                        b_show_centroid=False,
                        b_show_bbox3d=False,
                        b_show_info=False,
                        b_show_polygen=True)

save_img_filepath = os.path.join(os.path.dirname(bbox_filepath), f'{scene_name}_sem.png')
Image.fromarray(out_img).save(save_img_filepath)

# save synthetic object and room_layout as ply
scene_bbox_ply_fname = f'{scene_name}.ply'
scene_bbox_ply_filepath = os.path.join(os.path.dirname(bbox_filepath), scene_bbox_ply_fname)
scene_mesh = vis_scene_mesh(room_layout_mesh=None,
                            obj_bbox_lst=(wall_dict_lst + obj_bbox_dict_lst),
                            color_to_labels=COLOR_TO_ADEK_LABEL,
                            room_layout_bbox=None)
scene_mesh.export(scene_bbox_ply_filepath)
