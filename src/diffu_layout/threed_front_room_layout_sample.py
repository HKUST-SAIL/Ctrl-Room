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

import numpy as np
import torch as th
import torch.distributed as dist
from PIL import Image
from typing import List
import trimesh
import open3d as o3d

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

from dataset.threed_front_dataset_diffuscene import ThreedFrontDataset, ROOM_TYPE_DICT
from dataset.threed_front.threed_future_dataset import ThreedFutureDataset


from dataset.threed_front.metadata import TDFRONT_COLOR_TO_ADEK_LABEL
from utils.equirect_projection import vis_objs3d, vis_floor_ceiling_simple
from preprocess.prepare_st3d_dataset import vis_scene_mesh
from utils.utils import load_config

import seaborn as sns


def render_top2down(scene, renderables, color, mode, frame_path=None):
    if color is not None:
        try:
            color[0][0]
        except TypeError:
            color = [color] * len(renderables)
    else:
        color = [None] * len(renderables)

    scene.clear()
    for r, c in zip(renderables, color):
        if isinstance(r, Mesh) and c is not None:
            r.mode = mode
            r.colors = c
        scene.add(r)
    scene.render()
    if frame_path is not None:
        save_frame(frame_path, scene.frame)

    return np.copy(scene.frame)


def get_textured_objects(bbox_params_t: np.ndarray,
                         objects_dataset: ThreedFutureDataset,
                         classes: List,
                         diffusion=True,
                         no_texture=True):
    # For each one of the boxes replace them with an object
    renderables = []
    trimesh_meshes = []
    model_jids = []
    if diffusion:
        start, end = 0, bbox_params_t.shape[0]
    else:
        #for autoregressive model, we delete the 'start' and 'end'
        start, end = 1, bbox_params_t.shape[0] - 1

    color_palette = np.array(sns.color_palette('hls', len(classes) - 1))

    for j in range(start, end):
        query_size = bbox_params_t[j, -4:-1]
        query_label = classes[bbox_params_t[j, :-7].argmax(-1)]
        furniture = objects_dataset.get_closest_furniture_to_box(query_label, query_size)

        # Load the furniture and scale it as it is given in the dataset
        if no_texture:
            class_index = bbox_params_t[j, :-7].argmax(-1)
            raw_mesh = Mesh.from_file(furniture.raw_model_path, color=color_palette[class_index, :])
        else:
            raw_mesh = TexturedMesh.from_file(furniture.raw_model_path)
        raw_mesh.scale(furniture.scale)

        # Compute the centroid of the vertices in order to match the
        # bbox (because the prediction only considers bboxes)
        bbox = raw_mesh.bbox
        centroid = (bbox[0] + bbox[1]) / 2

        # Extract the predicted affine transformation to position the
        # mesh
        translation = bbox_params_t[j, -7:-4]
        theta = bbox_params_t[j, -1]
        R = np.zeros((3, 3))
        R[0, 0] = np.cos(theta)
        R[0, 2] = -np.sin(theta)
        R[2, 0] = np.sin(theta)
        R[2, 2] = np.cos(theta)
        R[1, 1] = 1.

        # Apply the transformations in order to correctly position the mesh
        raw_mesh.affine_transform(t=-centroid)
        raw_mesh.affine_transform(R=R, t=translation)
        renderables.append(raw_mesh)

        # Create a trimesh object for the same mesh in order to save
        # everything as a single scene
        tr_mesh = trimesh.load(furniture.raw_model_path, force="mesh")
        if no_texture:
            color = color_palette[class_index, :]
            tr_mesh.visual.vertex_colors = (color[None, :].repeat(tr_mesh.vertices.shape[0], axis=0).reshape(-1, 3) *
                                            255.0).astype(np.uint8)
            tr_mesh.visual.face_colors = (color[None, :].repeat(tr_mesh.faces.shape[0], axis=0).reshape(-1, 3) *
                                          255.0).astype(np.uint8)
        else:
            tr_mesh.visual.material.image = Image.open(furniture.texture_image_path)
            tr_mesh.visual.vertex_colors = (tr_mesh.visual.to_color()).vertex_colors[:, 0:3]
            print('convert texture to vertex colors')
        tr_mesh.vertices *= furniture.scale
        tr_mesh.vertices -= centroid
        tr_mesh.vertices[...] = tr_mesh.vertices.dot(R) + translation
        trimesh_meshes.append(tr_mesh)
        model_jids.append((furniture.raw_model_path).split('/')[-2])

    return renderables, trimesh_meshes, model_jids


def recover_scene_from_bbox_params(
    bbox_params: np.ndarray,
    dataset: ThreedFrontDataset,
):

    # recover oobject bounding box
    centroid_idx = len(dataset.class_labels)
    size_idx = centroid_idx + 3
    angle_idx = size_idx + 3
    obj_bbox_dict_lst = []
    for i in range(len(bbox_params)):
        obj_dict = {}
        # recover class label
        class_label_prob = bbox_params[i][:centroid_idx]
        # print(f'class_label_prob: {class_label_prob}')
        class_label_prob = np.where(class_label_prob > 0.5, 1, 0)
        class_label = dataset.class_labels[class_label_prob.argmax()]
        if class_label == 'end':
            continue
        obj_dict['class'] = class_label
        obj_center = bbox_params[i][centroid_idx:centroid_idx + 3]
        obj_dict['center'] = obj_center.tolist()
        obj_size = bbox_params[i][size_idx:size_idx + 3] * 2
        obj_dict['size'] = obj_size.tolist()
        obj_angle = bbox_params[i][angle_idx:angle_idx + 1]
        obj_dict['angles'] = [0, 0, obj_angle[0]]

        obj_bbox_dict_lst.append(obj_dict)

    scene_bbox_mesh = vis_scene_mesh(room_layout_mesh=None,
                                     obj_bbox_lst=obj_bbox_dict_lst,
                                     color_to_labels=TDFRONT_COLOR_TO_ADEK_LABEL,
                                     room_layout_bbox=None)
    return scene_bbox_mesh


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    log_dir = os.path.join(args.log_dir, datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f"))
    logger.configure(dir=log_dir, format_strs=['tensorboard', 'stdout', 'log', 'csv'])

    if args.room_type == 'bedroom':
        room_type = 'bedroom'
    elif args.room_type == 'livingroom':
        room_type = 'living room'
    elif args.room_type == 'diningroom':
        room_type = 'dining room'

    config = load_config(args.config_file)
    dataset = ThreedFrontDataset(config=config, room_type=room_type, is_train=False, is_test=True)
    logger.log("loaded {} from test dataset".format(len(dataset)))

    # Build the dataset of 3D models
    threed_furture_dataset = ThreedFutureDataset.from_pickled_dataset(args.path_to_pickled_3d_futute_models)
    print("Loaded {} 3D-FUTURE models".format(len(threed_furture_dataset)))

    logger.log("creating UNet model and diffusion model ...")
    model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
    model.to(dist_util.dev())
    model.eval()

    layout_channel_size = args.layout_channels
    layout_size = args.layout_size
    logger.log("sampling layout...")
    all_layout_lst = []
    all_layout_type_lst = []

    cond_text_prompt_lst = []
    scene_names_lst = []

    sample_result_folder = os.path.join(logger.get_dir(), f'{args.room_type}')
    os.makedirs(sample_result_folder, exist_ok=True)

    # Create the scene and the behaviour list for simple-3dviz top-down orthographic rendering, the arguments are same as preprocess_data.py
    # if args.render_top2down:
    #     scene_top2down = Scene(size=(256, 256), background=[0, 0, 0, 1])
    #     scene_top2down.up_vector = (0, 0, -1)
    #     scene_top2down.camera_target = (0, 0, 0)
    #     scene_top2down.camera_position = (0, 4, 0)
    #     scene_top2down.light = (0, 4, 0)
    #     scene_top2down.camera_matrix = Matrix44.orthogonal_projection(left=-3.1,
    #                                                                   right=3.1,
    #                                                                   bottom=3.1,
    #                                                                   top=-3.1,
    #                                                                   near=0.1,
    #                                                                   far=6)

    while len(all_layout_lst) * args.batch_size < args.num_samples:
        begin_tms = time.time()
        model_kwargs = {}
        if args.b_class_cond:
            # ignore 'undefined' class
            max_layout_types = (NUM_CLASSES - 1)
            layout_type_lst = th.randint(low=0, high=max_layout_types, size=(args.batch_size,), device=dist_util.dev())
            layout_type_lst = th.full((args.batch_size,), ROOM_TYPE_DICT[room_type], device=dist_util.dev())
            # model_kwargs["y"] = layout_type_lst
            scene_names_lst.append(f'{args.room_type}_{len(scene_names_lst)}')
        if args.b_text_cond:
            cond_data_lst = []
            for i in range(args.batch_size):
                scene_idx = np.random.choice(len(dataset))
                gt_scene, gt_cond_dict, scene_name = dataset[scene_idx]
                # text prompt from eval dataset
                cond_data_lst.append(gt_cond_dict['text_condition'])
                text_prompt = gt_cond_dict['text']
                cond_text_prompt_lst.append(text_prompt)
                scene_names_lst.append(scene_name)

                logger.log('text_prompt: {}'.format(text_prompt))

                # # debug gt scene
                # gt_scene_bbox_params = dataset.post_process(gt_scene.transpose(1, 0))
                # gt_scene_mesh = recover_scene_from_bbox_params(gt_scene_bbox_params, dataset)
                # gt_scene_bbox_ply_fname = f'{scene_name}_gt_bbox.ply'
                # gt_scene_mesh.export(os.path.join(sample_result_folder, gt_scene_bbox_ply_fname))

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

    arr = np.concatenate(all_layout_lst, axis=0)
    arr = arr[:args.num_samples]
    samples_arr = np.transpose(arr, (0, 2, 1))

    if args.b_text_cond:
        cond_text_prompt_lst = cond_text_prompt_lst[:args.num_samples]
    if args.b_class_cond:
        label_arr = np.concatenate(all_layout_type_lst, axis=0)
        label_arr = label_arr[:args.num_samples]

    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in samples_arr.shape])
        out_path = os.path.join(sample_result_folder, f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.b_class_cond:
            np.savez(out_path, samples_arr, label_arr)
        else:
            np.savez(out_path, samples_arr)

    for idx, scene_name in enumerate(scene_names_lst):
        # post process each sample
        scene_sample_result = samples_arr[idx]
        scene_sample_label = args.room_type
        print(f'scene_sample_label: {scene_sample_label}')

        bbox_params = dataset.post_process(scene_sample_result)
        logger.log(f'pose_processed bbox_params: {bbox_params.shape}')

        # recover oobject bounding box
        centroid_idx = len(dataset.class_labels)
        size_idx = centroid_idx + 3
        angle_idx = size_idx + 3
        obj_bbox_dict_lst = []
        for i in range(len(bbox_params)):
            obj_dict = {}
            # recover class label
            class_label_prob = bbox_params[i][:centroid_idx]
            # print(f'class_label_prob: {class_label_prob}')
            class_label_prob = np.where(class_label_prob > 0.5, 1, 0)
            class_label = dataset.class_labels[class_label_prob.argmax()]
            if class_label == 'end':
                continue
            obj_dict['class'] = class_label
            obj_center = bbox_params[i][centroid_idx:centroid_idx + 3]
            obj_dict['center'] = obj_center.tolist()
            obj_size = bbox_params[i][size_idx:size_idx + 3] * 2
            obj_dict['size'] = obj_size.tolist()
            obj_angle = bbox_params[i][angle_idx:angle_idx + 1]
            obj_dict['angles'] = [0, 0, obj_angle[0]]

            obj_bbox_dict_lst.append(obj_dict)

        logger.log(f'room {scene_name} have : {len(obj_bbox_dict_lst)} furniture')
        # save synthesis results as image
        out_img = np.zeros((512, 1024, 3), np.uint8)
        cam_position = np.zeros((3,), np.float32)
        out_img = vis_objs3d(out_img,
                             v_bbox3d=obj_bbox_dict_lst,
                             camera_position=cam_position,
                             color_to_labels=TDFRONT_COLOR_TO_ADEK_LABEL,
                             b_show_axes=False,
                             b_show_centroid=False,
                             b_show_bbox3d=True,
                             b_show_info=True,
                             b_show_polygen=False)

        curr_sample_folder = os.path.join(sample_result_folder, f'{idx}')
        os.makedirs(curr_sample_folder, exist_ok=True)
        save_img_filepath = os.path.join(curr_sample_folder, f'{scene_name}_sem.png')
        Image.fromarray(out_img).save(save_img_filepath)

        if args.b_text_cond:
            # save text prompt
            text_prompt_path = os.path.join(curr_sample_folder, f"text_prompt.txt")
            with open(text_prompt_path, 'w') as f:
                f.write(f'{cond_text_prompt_lst[idx]}\n')

        # save synthetic object and room_layout as ply
        scene_bbox_ply_fname = f'{scene_name}_bbox.ply'
        scene_bbox_ply_filepath = os.path.join(curr_sample_folder, scene_bbox_ply_fname)
        scene_bbox_mesh = vis_scene_mesh(room_layout_mesh=None,
                                         obj_bbox_lst=obj_bbox_dict_lst,
                                         color_to_labels=TDFRONT_COLOR_TO_ADEK_LABEL,
                                         room_layout_bbox=None)
        # scene_bbox_mesh.export(scene_bbox_ply_filepath)
        o3d_scene_bbox = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(scene_bbox_mesh.vertices),
                                                   triangles=o3d.utility.Vector3iVector(scene_bbox_mesh.faces))
        o3d_scene_bbox.vertex_normals = o3d.utility.Vector3dVector(scene_bbox_mesh.vertex_normals)
        o3d_scene_bbox.compute_triangle_normals()
        o3d_scene_bbox.vertex_colors = o3d.utility.Vector3dVector(scene_bbox_mesh.visual.vertex_colors[:, :3] / 255.0)
        o3d.io.write_triangle_mesh(scene_bbox_ply_filepath, o3d_scene_bbox)

        # save bbox params as npz
        scene_bbox_params_fname = f'{scene_name}_bbox_params.npz'
        scene_bbox_params_filepath = os.path.join(curr_sample_folder, scene_bbox_params_fname)
        np.savez(scene_bbox_params_filepath, bbox_params)

        # # search 3D-FUTURE models for objects
        # renderables_lst, trimesh_meshes_lst, model_jids = get_textured_objects(bbox_params,
        #                                                                        threed_furture_dataset,
        #                                                                        dataset.class_labels,
        #                                                                        diffusion=True,
        #                                                                        no_texture=True)
        # if args.render_top2down:
        #     path_to_image = "{}/{}.png".format(curr_sample_folder, scene_name)
        #     render_top2down(
        #         scene_top2down,
        #         renderables_lst,
        #         color=None,
        #         mode="shading",
        #         frame_path=path_to_image,
        #     )
        # # Create a trimesh scene and export it
        # path_to_scene_mesh = os.path.join(curr_sample_folder, "scene_mesh.ply")
        # whole_scene_mesh = trimesh.util.concatenate(trimesh_meshes_lst)
        # whole_scene_mesh.export(path_to_scene_mesh)

        # # export furniture mesh
        # scene_mesh_ply_fname = f'{scene_name}_mesh.ply'
        # scene_mesh_ply_filepath = os.path.join(curr_sample_folder, scene_mesh_ply_fname)
        # objects_mesh = trimesh.util.concatenate(objects_mesh_lst)
        # o3d_mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(objects_mesh.vertices),
        #                                      triangles=o3d.utility.Vector3iVector(objects_mesh.faces))
        # o3d_mesh.vertex_normals = o3d.utility.Vector3dVector(objects_mesh.vertex_normals)
        # o3d_mesh.compute_triangle_normals()
        # o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(objects_mesh.visual.vertex_colors[:, :3] / 255.0)
        # o3d.io.write_triangle_mesh(scene_mesh_ply_filepath, o3d_mesh)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        log_dir='sample_results',
        clip_denoised=True,
        num_samples=10,
        batch_size=1,
        use_ddim=False,
        model_path="",
        room_type='bedroom',
        path_to_pickled_3d_futute_models=
        "/mnt/nas_3dv/hdd1/datasets/3D_FRONT_FUTURE/threed_future_model_bedroom.pkl",  # bedroom furniture models
        config_file="../config/3dfront_bedroom_config.yaml",
        render_top2down=True,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
