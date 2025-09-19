#
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
#
"""Script for computing the FID score between real and synthesized scenes.
"""
import argparse
import os
import sys

sys.path.append('.')
sys.path.append('..')

import torch

import numpy as np
from PIL import Image

from cleanfid import fid

import shutil


def main(argv):
    parser = argparse.ArgumentParser(description=("Compute the FID scores between the real and the "
                                                  "synthetic images"))
    parser.add_argument(
        "--path_to_real_renderings",
        type=str,
        help="Path to the folder containing the real renderings",
        # default="/mnt/nas_3dv/hdd1/datasets/3D_FRONT_FUTURE/holistic_layout_20231113/livingroom_topdown_renders/")
        default="/mnt/nas_3dv/hdd1/fangchuan/eccv2024_other_methods/diffuscene_exps/test_livingroom_real")
    parser.add_argument(
        "--path_to_synthesized_renderings",
        type=str,
        help="Path to the folder containing the synthesized",
        # default="/mnt/nas_3dv/hdd1/fangchuan/originalDiffuScene/DiffuScene/scripts/3dfront_livingroom/diffusion_livingrooms_dim512_nomask_instancond_objfeats_lat32/gen_top2down_notexture_nofloor/"
        default="/mnt/nas_3dv/hdd1/fangchuan/eccv2024_other_methods/DiffuScene/pretrained/diningrooms_bert/gen_top2down_notexture_nofloor/"
    )

    args = parser.parse_args(argv)

    if not os.path.exists(args.path_to_real_renderings):
        print("Path to synthesized renderings does not exist! Create folder and copy all the rendering images there.")
        os.makedirs(args.path_to_real_renderings)

        dataset_folderpath = '/mnt/nas_3dv/hdd1/datasets/3D_FRONT_FUTURE/holistic_layout_20231113/threed_front_diningroom/'
        test_splits_filepath = '/mnt/nas_3dv/hdd1/datasets/3D_FRONT_FUTURE/holistic_layout_20231113/splits/diningroom_test.lst'
        train_splits_filepath = test_splits_filepath.replace('test', 'train')
        valid_splits_filepath = test_splits_filepath.replace('test', 'val')

        test_splits = []
        with open(test_splits_filepath, 'r') as f:
            for line in f:
                test_splits.append(line.strip())
        with open(train_splits_filepath, 'r') as f:
            for line in f:
                test_splits.append(line.strip())
        with open(valid_splits_filepath, 'r') as f:
            for line in f:
                test_splits.append(line.strip())
        real_images_path_lst = [
            os.path.join(dataset_folderpath, f, 'rendered_scene_notexture_256.png')
            for f in os.listdir(dataset_folderpath)
            if f.split('_')[-1] in test_splits
        ]

        print(f"Copying real images to {args.path_to_real_renderings}...")
        for i, img_path in enumerate(real_images_path_lst):
            shutil.copyfile(img_path, "{}/{:05d}.png".format(args.path_to_real_renderings, i))

    path_to_test_real = args.path_to_real_renderings
    real_images_path_lst = [
        os.path.join(path_to_test_real, f) for f in os.listdir(path_to_test_real) if f.endswith(".png")
    ]
    # Number of images to be copied
    N = len(real_images_path_lst)
    print("Number of real images: {}".format(len(real_images_path_lst)))

    fake_img_folderpath = args.path_to_synthesized_renderings
    fake_images_path_lst = [
        os.path.join(fake_img_folderpath, f) for f in os.listdir(fake_img_folderpath) if f.endswith(".png")
    ]
    print("Generating temporary a folder with test_fake images...")
    path_to_test_fake = "/mnt/nas_3dv/hdd1/fangchuan/eccv2024_other_methods/diffuscene_exps/test_diningroom_fake"
    if not os.path.exists(path_to_test_fake):
        os.makedirs(path_to_test_fake)
    print("Number of fake images: {}".format(len(fake_images_path_lst)))
    assert len(fake_images_path_lst) > N

    scores = []
    for _ in range(10):
        np.random.shuffle(fake_images_path_lst)
        synthesized_images_subset = np.random.choice(fake_images_path_lst, N)
        for i, fi in enumerate(synthesized_images_subset):
            shutil.copyfile(fi, "{}/{:05d}.png".format(path_to_test_fake, i))

        # Compute the FID score
        fid_score = fid.compute_fid(path_to_test_real, path_to_test_fake, device=torch.device("cpu"))
        scores.append(fid_score)
        print(fid_score)
    print(sum(scores) / len(scores))
    print(np.std(scores))


if __name__ == "__main__":
    main(None)
