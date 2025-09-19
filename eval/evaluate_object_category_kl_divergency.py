"""Script used for estimating the KL-divergence between the object categories
of real and generated scenes."""
import os
import sys
sys.path.append('.')
sys.path.append('..')

import argparse

import numpy as np
import torch
import torch.distributed as dist

from tqdm import tqdm

from improved_diffusion import logger, dist_util
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from dataset.st3d_dataset import ST3DDataset
from dataset.threed_front_dataset import ThreedFrontDataset
from dataset.metadata import ST3D_BEDROOM_FURNITURE, ST3D_LIVINGROOM_FURNITURE, ST3D_DININGROOM_FURNITURE
from dataset.threed_front.metadata import THREED_FRONT_BEDROOM_FURNITURE, THREED_FRONT_LIVINGROOM_FURNITURE, THREED_FRONT_DININGROOM_FURNITURE
import datetime


def categorical_kl(p, q):
    return (p * (np.log(p + 1e-6) - np.log(q + 1e-6))).sum()


def main():
    b_skip_synthetic = True

    args = create_argparser().parse_args()
    dist_util.setup_dist()

    # set up logger
    log_dir = os.path.join(args.log_dir, datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f"))
    logger.configure(dir=log_dir, format_strs=['tensorboard', 'stdout', 'log', 'csv'])

    if not b_skip_synthetic:
        logger.log("creating UNet model and diffusion model ...")
        model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
        model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
        model.to(dist_util.dev())
        model.eval()

    if args.room_type == 'bedroom':
        room_type = 'bedroom'
    elif args.room_type == 'livingroom':
        room_type = 'living room'
    elif args.room_type == 'diningroom':
        room_type = 'dining room'

    if args.dataset_type == "3d_front":
        dataset = ThreedFrontDataset(root_dir=args.dataset_dir,
                                 room_type=room_type,
                                 is_train=False,
                                 is_test=True,
                                 max_text_sentences=4)
    elif args.dataset_type == "st3d":
        dataset = ST3DDataset(root_dir=args.dataset_dir, flip=False, rotate=False, gamma=False, return_path=True)

    # Generate synthetic rooms with the pre-trained model
    layout_channel_size = args.layout_channels
    layout_size = args.layout_size
    class_label_dim = layout_channel_size - 3 - 3 - 2
    center_dim = 3
    size_dim = 3
    angle_dim = 2

    logger.log("sampling layout...")

    ground_truth_scenes_lst = []
    ground_truth_scenes_type_lst = []

    synthesized_scenes_type_lst = []
    synthesized_scenes_lst = []
    # while len(synthesized_scenes_lst) * args.batch_size < args.num_samples:
    for i in tqdm(range(args.num_samples)):
        scene_idx = np.random.choice(len(dataset))
        gt_scene, gt_scene_type, scene_name = dataset[scene_idx]
        ground_truth_scenes_lst.append(gt_scene)
        ground_truth_scenes_type_lst.append(gt_scene_type['y'])

        if not b_skip_synthetic:
            batch_size = args.batch_size
            model_kwargs = {}
            if args.b_class_cond:
                # ignore 'undefined' class
                max_layout_types = (NUM_CLASSES - 1)
                layout_type_lst = torch.randint(low=0,
                                                high=max_layout_types,
                                                size=(batch_size,),
                                                device=dist_util.dev())
                layout_type_lst = torch.full((batch_size,), 2, device=dist_util.dev())
                model_kwargs["y"] = layout_type_lst
            sample_fn = (diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop)
            sample = sample_fn(
                model=model,
                shape=(batch_size, layout_channel_size, layout_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )

            gathered_samples = [torch.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            synthesized_scenes_lst.extend([sample.cpu().numpy() for sample in gathered_samples])
            if args.b_class_cond:
                gathered_labels = [torch.zeros_like(layout_type_lst) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_labels, layout_type_lst)
                synthesized_scenes_type_lst.extend([labels.cpu().numpy() for labels in gathered_labels])
            logger.log(f"created {len(synthesized_scenes_lst) * batch_size} samples")

    if b_skip_synthetic:
        samples_filepath = '../sample_results/openai-2023-09-22-22-54-20-584989/diningroom/samples_1000x45x36.npz'
        samples_result = np.load(samples_filepath)
        synthesized_scenes = samples_result['arr_0']
    else:
        # num x 32 x 23
        synthesized_scenes = np.concatenate(synthesized_scenes_lst, axis=0)
        synthesized_scenes = synthesized_scenes[:args.num_samples]
    # synthesized_scenes = np.transpose(synthesized_scenes, (0, 2, 1))
    logger.info(f"synthesized_scenes.shape: {synthesized_scenes.shape}")

    if not b_skip_synthetic:
        # save the synthesized scenes
        if args.b_class_cond:
            label_arr = np.concatenate(synthesized_scenes_type_lst, axis=0)
            label_arr = label_arr[:args.num_samples]
        if dist.get_rank() == 0:
            shape_str = "x".join([str(x) for x in synthesized_scenes.shape])
            out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
            logger.log(f"saving to {out_path}")
            if args.b_class_cond:
                np.savez(out_path, synthesized_scenes, label_arr)
            else:
                np.savez(out_path, synthesized_scenes)
        dist.barrier()
        logger.log("sampling complete")

    ground_truth_scenes = np.stack(ground_truth_scenes_lst, axis=0)
    ground_truth_scenes = ground_truth_scenes[:args.num_samples]
    ground_truth_scenes = np.transpose(ground_truth_scenes, (0, 2, 1))
    logger.info(f"ground_truth_scenes.shape: {ground_truth_scenes.shape}")

    # Firstly compute the frequencies of the class labels
    # TODO: should skip empty!
    def discard_empty_objects(scenes: np.ndarray, scene_type: str):
        new_scenes = []
        if scene_type == 'bedroom':
            class_labels_lst = (ST3D_BEDROOM_FURNITURE) if args.dataset_type == 'st3d' else THREED_FRONT_BEDROOM_FURNITURE
        elif scene_type == 'living room':
            class_labels_lst = (ST3D_LIVINGROOM_FURNITURE) if args.dataset_type == 'st3d' else THREED_FRONT_LIVINGROOM_FURNITURE
        elif scene_type == 'dining room':
            class_labels_lst = (ST3D_DININGROOM_FURNITURE) if args.dataset_type == 'st3d' else THREED_FRONT_DININGROOM_FURNITURE
        else:
            raise NotImplementedError

        B, C, W = scenes.shape
        for b_idx in range(B):
            new_scene = []
            for c_idx in range(C):
                object = scenes[b_idx, c_idx, :]
                class_label_prob = object[:class_label_dim]
                class_label_prob = np.where(class_label_prob > 0.5, 1, 0)
                object[:class_label_dim] = class_label_prob
                class_label = class_labels_lst[class_label_prob.argmax()]
                if class_label == 'empty':
                    continue
                else:
                    new_scene.append(object)
            # print(f'new_scene shape: {np.stack(new_scene, axis=0).shape}')
            new_scenes.append(np.stack(new_scene, axis=0))
        return np.array(new_scenes)

    ground_truth_scenes = discard_empty_objects(ground_truth_scenes, room_type)
    synthesized_scenes = discard_empty_objects(synthesized_scenes, room_type)

    gt_class_labels = sum([d[:, :class_label_dim - 1].sum(0) for d in ground_truth_scenes]) / sum(
        [d[:, :class_label_dim - 1].shape[0] for d in ground_truth_scenes])
    syn_class_labels = sum([d[:, :class_label_dim - 1].sum(0) for d in synthesized_scenes]) / sum(
        [d[:, :class_label_dim - 1].shape[0] for d in synthesized_scenes])

    print(f'gt_class_labels.shape: {gt_class_labels.shape}')
    print(f'gt_class_labels: {gt_class_labels.sum()}')
    print(f'syn_class_labels.shape: {syn_class_labels.shape}')
    print(f'syn_class_labels: {syn_class_labels.sum()}')
    assert 0.9999 <= gt_class_labels.sum() <= 1.01
    assert 0.9999 <= syn_class_labels.sum() <= 1.01
    stats = {}
    stats["class_labels"] = categorical_kl(gt_class_labels, syn_class_labels)
    logger.info(stats)
    stats_filepath = os.path.join(args.log_dir, "kl_divergency_stats.npz")

    dataset_class_labels = {
        'bedroom': ST3D_BEDROOM_FURNITURE if args.dataset_type == 'st3d' else THREED_FRONT_BEDROOM_FURNITURE,
        'livingroom': ST3D_LIVINGROOM_FURNITURE if args.dataset_type == 'st3d' else THREED_FRONT_LIVINGROOM_FURNITURE,
        'diningroom': ST3D_DININGROOM_FURNITURE if args.dataset_type == 'st3d' else THREED_FRONT_DININGROOM_FURNITURE,
    }[args.room_type]
    for c, gt_cp, syn_cp in zip(dataset_class_labels, gt_class_labels, syn_class_labels):
        logger.info("{}: target: {} / synth: {}".format(c, gt_cp, syn_cp))

    # gt_cooccurrences = np.zeros((len(classes) - 2, len(classes) - 2))
    # syn_cooccurrences = np.zeros((len(classes) - 2, len(classes) - 2))
    # for gt_scene, syn_scene in zip(ground_truth_scenes, synthesized_scenes):
    #     gt_classes = gt_scene["class_labels"].argmax(axis=-1)
    #     syn_classes = syn_scene["class_labels"][1:-1].argmax(axis=-1)

    #     for ii in range(len(gt_classes)):
    #         r = gt_classes[ii]
    #         for jj in range(ii + 1, len(gt_classes)):
    #             c = gt_classes[jj]
    #             gt_cooccurrences[r, c] += 1

    #     for ii in range(len(syn_classes)):
    #         r = syn_classes[ii]
    #         for jj in range(ii + 1, len(syn_classes)):
    #             c = syn_classes[jj]
    #             syn_cooccurrences[r, c] += 1

    # print("Saving stats at {}".format(path_to_stats))
    # np.savez(path_to_stats,
    #          stats=stats,
    #          classes=classes,
    #          gt_class_labels=gt_class_labels,
    #          syn_class_labels=syn_class_labels,
    #          gt_cooccurrences=gt_cooccurrences,
    #          syn_cooccurrences=syn_cooccurrences)


def create_argparser():
    defaults = dict(
        dataset_type="3d_front",
        dataset_dir='/mnt/nas_3dv/hdd1/datasets/3D_FRONT_FUTURE/test/bedroom/',
        log_dir='sample_results',
        clip_denoised=True,
        num_samples=1000,
        batch_size=1,
        use_ddim=False,
        model_path="",
        room_type='bedroom',
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
