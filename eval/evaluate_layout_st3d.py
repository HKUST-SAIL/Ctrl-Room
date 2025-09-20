# this script is used to evaluate the layout synthesis results on the Structured3D dataset
# 1. average IoU among furniture;
# 2. layout consistency to the input text: furniture types, the number of walls
import sys
sys.path.append('.')
sys.path.append('..')

import os
import json
import argparse
from glob import glob
from collections import Counter, OrderedDict

import torch
from num2words import num2words
import numpy as np
from dataset.metadata import ST3D_BEDROOM_FURNITURE, \
                            ST3D_LIVINGROOM_FURNITURE, \
                            ST3D_STUDY_FURNITURE, \
                            ST3D_KITCHEN_FURNITURE, \
                            ST3D_BATHROOM_FURNITURE
from dataset.metadata import COLOR_TO_ADEK_LABEL
from src.diffu_layout.improved_diffusion.losses import axis_aligned_bbox_overlaps_3d
from utils.utils import my_compute_box_3d
from utils.utils import vis_scene_mesh
import open3d as o3d

def extract_wall_num_from_text(text_prompt):
    # the input text prompt is like "The room has four walls. xxxxx."
    sentences = text_prompt.split('.')
    pos = sentences[0].find('walls')
    if pos == -1:
        return 'zero'
    first_sentence = sentences[0].split(' ')
    return first_sentence[-2]

def extract_furniture_cats_from_text(text_prompt, room_type:str='livingroom'):
    # the input text prompt is like "The study has four walls.The room has a window , a chair and a picture .There is a desk to the left of the chair .There is a lamp above the chair ."
    if room_type == 'livingroom':
        # skip the last two categories
        furniture_cats = ST3D_LIVINGROOM_FURNITURE[:-2]
    elif room_type == 'bedroom':
        furniture_cats = ST3D_BEDROOM_FURNITURE[:-2]
    elif room_type == 'study':
        furniture_cats = ST3D_STUDY_FURNITURE[:-2]
    elif room_type == 'kitchen':
        furniture_cats = ST3D_KITCHEN_FURNITURE[:-2]
    elif room_type == 'bathroom':
        furniture_cats = ST3D_BATHROOM_FURNITURE[:-2]
    else:
        raise ValueError(f'Unknown room type: {room_type}')
    
    # extract furniture categories from the text prompt
    return_cats = []
    # sentences = text_prompt.split('.')
    # for sentence in sentences:
    #     sentence = sentence.lower()
    #     for cat in furniture_cats:
    #         times = sentence.count(cat)
    #         if times > 0:
    #             return_cats.append([cat] * times)
    
    for cat in furniture_cats:
        if cat in text_prompt:
            return_cats.append([cat])
    return return_cats

def load_prompts(text_prompts_filepath):
    prompts_lst = []
    with open(text_prompts_filepath, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            prompts_lst.append(line.strip())
    return prompts_lst

def computer_intersection(objects_dict, judge_mesh_intersec=False, room_dir:str='', debug=True):
    if debug:
        scene_mesh = vis_scene_mesh(room_layout_mesh=None,
                            obj_bbox_lst=objects_dict,
                            color_to_labels=COLOR_TO_ADEK_LABEL,
                            room_layout_bbox=None)
        corners_ply = o3d.geometry.PointCloud()
    box_list = []
    for i in range(len(objects_dict)):
        box_center = np.array(objects_dict[i]['center'])
        box_size = np.array(objects_dict[i]['size'])
        box_angles = np.array(objects_dict[i]['angles'])
        box_corners = my_compute_box_3d(box_center, box_size / 2, -box_angles[2])
        left_bott_corner = box_corners[-1]
        right_top_corner = box_corners[1]
        box = np.concatenate([ left_bott_corner, right_top_corner], axis=-1)
        if debug:
            o3d_corners = o3d.geometry.PointCloud()
            o3d_corners.points = o3d.utility.Vector3dVector(np.array([ left_bott_corner, right_top_corner]))
            o3d_corners.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0]]*2))
            corners_ply += o3d_corners
        box_list.append(box)
    
    if debug:
        save_path = os.path.join(room_dir, 'scene_bbox.ply')
        scene_mesh.export(save_path)
        o3d.io.write_point_cloud(os.path.join(room_dir, 'bbox_corners.ply'), corners_ply)
        
    if len(box_list) >1:
        box_array = np.stack(box_list, axis=0).astype(np.float32)
    else:
        return len(objects_dict), 1, 0, 0, 0
    
    box_tensor = torch.from_numpy(box_array[None, ])
    box_iou, overlap_ratio = axis_aligned_bbox_overlaps_3d(box_tensor, box_tensor)
    
    box_iou = box_iou.squeeze(0).cpu().numpy()

    iou_list = []
    insec_list = []
    for i in range(len(objects_dict)):
        for j in range(i+1, len(objects_dict)):
            if box_iou[i, j] > 0.0:
                if judge_mesh_intersec:
                    s1, s2 = pv.wrap(trimeshes[i]), pv.wrap(trimeshes[j])
                    intersection, s1_split, s2_split = s1.intersection(s2)
                    if intersection.n_verts >0 and intersection.n_faces >0:
                        iou_list.append(box_iou[i, j])
                        insec_list.append(1)
                    else:
                        iou_list.append(0)
                        insec_list.append(0)
                else:
                    iou_list.append(box_iou[i, j])
                    insec_list.append(1)
            else:
                iou_list.append(0)
                insec_list.append(0)
    # return num_of_objects, number of pairs, avg iou (iou sum / pairs), avg intersection numbers ( intersec sum/ pairs)
    return len(objects_dict), len(iou_list), float(sum(iou_list))/len(iou_list), float(sum(insec_list))/len(iou_list), overlap_ratio.item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=("Compute the layout consistency and average IoU of synthetic layouts"))
    parser.add_argument(
        "--path_to_synthesized_layouts",
        type=str,
        help="Path to the folder containing the synthesized layouts",
        default="/mnt/nas_3dv/hdd1/fangchuan/eccv2024_ctrlroom/rebuttal/layout_eval_study"
    )
    parser.add_argument(
        "--path_to_output",
        type=str,
        help="Path to the folder containing the synthesized",
        default=
        "/mnt/nas_3dv/hdd1/fangchuan/eccv2024_ctrlroom/rebuttal/layout_eval"
    )

    args = parser.parse_args()

    synthesized_layout_dir = args.path_to_synthesized_layouts
    output_dir = args.path_to_output
    
    synthesized_roomtype_lst = [f for f in os.listdir(synthesized_layout_dir) if os.path.isdir(os.path.join(synthesized_layout_dir, f)) and f in ['bedroom', 'livingroom', 'study', 'kitchen', 'bathroom']]
    print(f'Found {len(synthesized_roomtype_lst)} synthesized room types')
    
    for room_type in synthesized_roomtype_lst:
        type_dir = os.path.join(synthesized_layout_dir, room_type)
        room_lst = [f for f in os.listdir(type_dir) if os.path.isdir(os.path.join(type_dir, f)) and f.isdigit()]
        room_lst = sorted(room_lst, key=lambda x: int(x))
        
        text_prompts_filepath = os.path.join(type_dir, 'text_prompt.txt')
        prompts_lst = load_prompts(text_prompts_filepath)
        
        wall_num_consistency = 0
        furniture_cat_consistency = 0
        furniture_ious = 0
        for idx, room in enumerate(room_lst):
            room_dir = os.path.join(type_dir, room)
            text_prompt = prompts_lst[idx]
            print(f'Processing {room_type}-{room}, prompt: {text_prompt}')
            
            layout_json_filepaths = glob(os.path.join(room_dir, '*.json'))
            if len(layout_json_filepaths) == 0:
                print(f'No layout json file found in {room_dir}')
                continue
            layout_json_filepath = layout_json_filepaths[0]
            scene_name = os.path.basename(layout_json_filepath).split('.')[0]
            # parse the layout json file
            object_bbox_lst = []
            wall_bbox_lst = []
            with open(layout_json_filepath) as f:
                layout_dicts = json.load(f)
                object_bbox_dicts = layout_dicts['objects']
                wall_bbox_dicts = layout_dicts['walls']

            # evaluate the wall numbers
            synthesized_wall_num = len(wall_bbox_dicts)
            print(f'Synthesized wall number: {synthesized_wall_num}')
            text_wall_num = extract_wall_num_from_text(text_prompt)
            print(f'Text wall number: {text_wall_num}')
            if num2words(synthesized_wall_num) == text_wall_num:
                wall_num_consistency += 1
                
            # evaluate furnite categories
            synthesized_furniture_cats = []
            for funiture in object_bbox_dicts:
                synthesized_furniture_cats.append([funiture['class']])            
            synthesized_furniture_counts = Counter(sum(synthesized_furniture_cats, []))
            synthesized_furniture_counts = OrderedDict(sorted(synthesized_furniture_counts.items(), key=lambda x: -x[1]))
            print(f'Synthesized furniture categories: {synthesized_furniture_counts}')
            text_furniture_cats = extract_furniture_cats_from_text(text_prompt, room_type)
            text_furniture_counts = Counter(sum(text_furniture_cats, []))
            text_furniture_counts = OrderedDict(sorted(text_furniture_counts.items(), key=lambda x: -x[1]))
            print(f'Text furniture categories: {text_furniture_counts}')
            # calculate the consistency of furniture categories
            cat_consistency = 0
            for cat, count in text_furniture_counts.items():
                if cat in synthesized_furniture_counts:
                    if synthesized_furniture_counts[cat] >= count:
                        cat_consistency += 1
            furniture_cat_consistency += cat_consistency/ len(text_furniture_counts)
            
            # evaluate the average IoU among furniture
            num_of_objects, num_of_pairs, avg_iou, avg_insec, overlap_ratio = computer_intersection(object_bbox_dicts, room_dir=room_dir, debug=True)
            print(f'Average IoU: {avg_iou}, average intersection ratio: {avg_insec}, overlap ratio: {overlap_ratio}')
            furniture_ious += avg_iou
            
        wall_consis_ratio = wall_num_consistency / len(room_lst)
        furniture_consis_ratio = furniture_cat_consistency / len(room_lst)
        furniture_ious = furniture_ious / len(room_lst)
        print(f'Room type: {room_type}, wall consistency: {wall_consis_ratio}, furniture consistency: {furniture_consis_ratio}, furniture IoU: {furniture_ious}')
        
