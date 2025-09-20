"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import os
import sys

sys.path.append(".")  # Adds higher directory to python modules path.
sys.path.append("..")  # Adds higher directory to python modules path.
import argparse
import datetime
import time
import json

import numpy as np
import torch as th
import torch.distributed as dist
from PIL import Image

from src.diffu_layout.improved_diffusion import dist_util, logger
from src.diffu_layout.improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

from dataset.st3d_dataset import ST3DDataset
from dataset.metadata import ST3D_BEDROOM_QUAD_WALL_MAX_LEN, ST3D_BEDROOM_FURNITURE, \
                            ST3D_LIVINGROOM_QUAD_WALL_MAX_LEN, ST3D_LIVINGROOM_FURNITURE,\
                            ST3D_DININGROOM_FURNITURE, ST3D_KITCHEN_FURNITURE, \
                            ST3D_STUDY_FURNITURE, ST3D_STUDY_QUAD_WALL_MAX_LEN, \
                            ST3D_BATHROOM_FURNITURE, ST3D_BATHROOM_QUAD_WALL_MAX_LEN, \
                            COLOR_TO_ADEK_LABEL

from utils.equirect_projection import vis_objs3d, vis_floor_ceiling_simple
from utils.utils import reconstrcut_floor_ceiling_from_quad_walls,vis_scene_mesh, euler_angle_to_matrix

TARGET_SCENE_NAMES = ['scene_03279_413']

def recover_quad_wall_layout_mesh(dataset_type: str,
                                  room_type: str,
                                  quad_wall_lst: np.ndarray,
                                  object_bbox_lst: np.ndarray,
                                  room_layout_bbox_size: np.array = np.array([1.0, 1.0, 1.0])):

    assert dataset_type == 'st3d'
    if room_type == 'bedroom':
        class_labels_lst = ST3D_BEDROOM_FURNITURE
    elif room_type == 'livingroom':
        class_labels_lst = ST3D_LIVINGROOM_FURNITURE
    elif room_type == 'diningroom':
        class_labels_lst = ST3D_DININGROOM_FURNITURE
    elif room_type == 'kitchen':
        class_labels_lst = ST3D_KITCHEN_FURNITURE
    elif room_type == 'study':
        class_labels_lst = ST3D_STUDY_FURNITURE
    elif room_type == 'bathroom':
        class_labels_lst = ST3D_BATHROOM_FURNITURE
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


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    log_dir = args.log_dir
    logger.configure(dir=log_dir, format_strs=['tensorboard', 'stdout', 'log', 'csv'])

    dataset = ST3DDataset(root_dir=args.data_dir, 
                          max_text_sentences=4, 
                          return_scene_name=True, 
                          random_text_desc=False, 
                          use_gpt_text_desc=args.use_gpt_text_desc,
                          train_stats_file=args.dataset_stats_file)

    logger.log("creating UNet model and diffusion model ...")
    model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
    model.to(dist_util.dev())
    model.eval()
    
    # uncomment to get deterministic generations
    th.manual_seed(12345) 

    layout_channel_size = args.layout_channels
    layout_size = args.layout_size
    # sample all test data
    if args.num_samples == -1:
        args.num_samples = len(dataset)

    logger.log("sampling layout...")
    all_layout_lst = []
    all_layout_type_lst = []

    cond_text_prompt_lst = []
    scene_names_lst = []
    while len(all_layout_lst) * args.batch_size < args.num_samples:
        begin_tms = time.time()
        model_kwargs = {}
        if args.b_class_cond:
            # ignore 'undefined' class
            max_layout_types = (NUM_CLASSES - 1)
            layout_type_lst = th.randint(low=0, high=max_layout_types, size=(args.batch_size,), device=dist_util.dev())
            layout_type_lst = th.full((args.batch_size,), 2, device=dist_util.dev())
            model_kwargs["y"] = layout_type_lst
        if args.b_text_cond:
            cond_data_lst = []
            if args.b_scene_name:
                for i, scene_name in enumerate(TARGET_SCENE_NAMES):
                    gt_scene, gt_cond_dict, gt_scene_name = dataset.get_scene(scene_name)
                    assert scene_name == gt_scene_name
                    # text prompt from eval dataset
                    cond_data_lst.append(gt_cond_dict['text_condition'])
                    text_prompt = gt_cond_dict['text']
                    cond_text_prompt_lst.append(text_prompt)
                    scene_names_lst.append(scene_name)
                    logger.log('text_prompt: {}'.format(text_prompt))
            else:
                for i in range(args.batch_size):
                    if args.num_samples < len(dataset):
                        scene_idx = np.random.choice(len(dataset))
                    else:
                        scene_idx = len(all_layout_lst) * args.batch_size + i
                    gt_scene, gt_cond_dict, scene_name = dataset[scene_idx]
                    # text prompt from eval dataset
                    cond_data_lst.append(gt_cond_dict['text_condition'])
                    text_prompt = gt_cond_dict['text']
                    cond_text_prompt_lst.append(text_prompt)
                    scene_names_lst.append(scene_name)
                    logger.log('text_prompt: {}'.format(text_prompt))

            model_kwargs["text_condition"] = th.from_numpy(np.stack(cond_data_lst)).to(dist_util.dev(), dtype=th.float32)

        sample_fn = (diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop)
        sample = sample_fn(
            model=model,
            shape=(args.batch_size, layout_channel_size, layout_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        # calc sampling time
        elaps_time = time.time() - begin_tms
        logger.log(f'sample shape: {sample.shape}')
        logger.log(f'sample time: {elaps_time}')

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_layout_lst.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.b_class_cond:
            gathered_labels = [th.zeros_like(layout_type_lst) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_labels, layout_type_lst)
            all_layout_type_lst.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_layout_lst) * args.batch_size} samples")

    samples_arr = np.concatenate(all_layout_lst, axis=0)
    samples_arr = samples_arr[:args.num_samples]
    samples_arr = np.transpose(samples_arr, (0, 2, 1))
    print(f'samples_arr.shape: {samples_arr.shape}')

    sample_result_folder = os.path.join(logger.get_dir(), f'{args.room_type}')
    if not os.path.exists(sample_result_folder):
        os.makedirs(sample_result_folder)

    cond_text_prompt_lst = cond_text_prompt_lst[:args.num_samples]
    if args.b_class_cond:
        label_arr = np.concatenate(all_layout_type_lst, axis=0)
        label_arr = label_arr[:args.num_samples]
    elif args.b_text_cond:
        text_prompt_path = os.path.join(sample_result_folder, f"text_prompt.txt")
        with open(text_prompt_path, 'w') as f:
            for i in range(args.num_samples):
                f.write(f'{cond_text_prompt_lst[i]}\n')


    # load sample results: BxNxChannel
    post_sample_arr = []
    for idx, scene_name in enumerate(scene_names_lst):
        scene_sample_result = samples_arr[idx]
        # print(f'scene_sample_result.shape: {scene_sample_result.shape}')
        # descale samples
        scene_sample_result = dataset.post_process(scene_sample_result)
        # print(f'scene_sample_result.shape: {scene_sample_result.shape}')
        # print(f'descaled scene_sample_result.shape: {scene_sample_result.shape}')
        post_sample_arr.append(scene_sample_result)

        if args.room_type == 'bedroom':
            max_wall_num = ST3D_BEDROOM_QUAD_WALL_MAX_LEN
        elif args.room_type == 'livingroom':
            max_wall_num = ST3D_LIVINGROOM_QUAD_WALL_MAX_LEN
        elif args.room_type == 'kitchen':
            max_wall_num = ST3D_BEDROOM_QUAD_WALL_MAX_LEN
        elif args.room_type == 'study':
            max_wall_num = ST3D_STUDY_QUAD_WALL_MAX_LEN
        elif args.room_type == 'bathroom':
            max_wall_num = ST3D_BATHROOM_QUAD_WALL_MAX_LEN
        else:
            raise NotImplementedError
        # quad walls
        quad_wall_lst = scene_sample_result[:max_wall_num, :]
        # objects
        obj_bbox_lst = scene_sample_result[max_wall_num:, :]

        wall_dict_lst, obj_bbox_dict_lst = recover_quad_wall_layout_mesh(dataset_type='st3d',
                                                                         room_type=args.room_type,
                                                                         quad_wall_lst=quad_wall_lst,
                                                                         object_bbox_lst=obj_bbox_lst,
                                                                         )

        # save synthesis results as image
        out_img = np.zeros((512, 1024, 3), np.uint8)
        cam_position = np.zeros((3,), np.float32)
        # post-process walls
        reconstrcut_floor_ceiling_from_quad_walls(quad_walls_lst=wall_dict_lst)
        out_img = vis_floor_ceiling_simple(image=out_img, color_to_labels=COLOR_TO_ADEK_LABEL)
        out_img = vis_objs3d(out_img,
                             v_bbox3d=(wall_dict_lst + obj_bbox_dict_lst),
                             camera_position=cam_position,
                             color_to_labels=COLOR_TO_ADEK_LABEL,
                             b_show_axes=False,
                             b_show_centroid=False,
                             b_show_bbox3d=False,
                             b_show_info=False,
                             b_show_polygen=True)

        curr_sample_folder = os.path.join(sample_result_folder, f'{idx}')
        os.makedirs(curr_sample_folder, exist_ok=True)
        save_img_filepath = os.path.join(curr_sample_folder, f'{scene_name}_sem.png')
        Image.fromarray(out_img).save(save_img_filepath)

        # save synthetic object and room_layout as ply
        scene_bbox_ply_fname = f'{scene_name}.ply'
        scene_bbox_ply_filepath = os.path.join(curr_sample_folder, scene_bbox_ply_fname)
        scene_mesh = vis_scene_mesh(room_layout_mesh=None,
                                    obj_bbox_lst=(wall_dict_lst + obj_bbox_dict_lst),
                                    color_to_labels=COLOR_TO_ADEK_LABEL,
                                    room_layout_bbox=None)
        scene_mesh.export(scene_bbox_ply_filepath)

        # scene layout file
        scene_layout_fname = f'{scene_name}.json'
        scene_layout_filepath = os.path.join(curr_sample_folder, scene_layout_fname)
        with open(scene_layout_filepath, 'w') as f:
            json.dump({'walls': wall_dict_lst, 'objects': obj_bbox_dict_lst}, f, indent=4)

    # save sample results
    post_sample_arr = np.array(post_sample_arr)
    print(f'post_sample_arr.shape: {post_sample_arr.shape}')
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in post_sample_arr.shape])
        out_path = os.path.join(sample_result_folder, f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, post_sample_arr)
    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        data_dir='data',
        log_dir='sample_results',
        clip_denoised=True,
        num_samples=10,
        batch_size=1,
        use_ddim=False,
        model_path="",
        room_type='bedroom',
        dataset_stats_file=None,
        use_gpt_text_desc=False,
        b_scene_name=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
