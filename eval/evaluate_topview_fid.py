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
        default="/mnt/nas_3dv/hdd1/datasets/3D_FRONT_FUTURE/holistic_layout_20231113/livingroom_topdown_renders/"
        # default=
        # "/mnt/nas_3dv/hdd1/fangchuan/HolisticDiffuScene/log/3dfront_livingroom/unconditional/sample_results/openai-2023-11-22-16-26-54-891915/livingroom/"
    )
    parser.add_argument(
        "--path_to_synthesized_renderings",
        type=str,
        help="Path to the folder containing the synthesized",
        # default= "/mnt/nas_3dv/hdd1/fangchuan/HolisticDiffuScene/sample_results/openai-2023-09-23-12-23-59-306819/topview/")
        default=
        "/mnt/nas_3dv/hdd1/fangchuan/HolisticDiffuScene/log/3dfront_livingroom/textconditional-0122/sample_results/openai-2024-01-27-08-16-04-293807/livingroom"
    )

    parser.add_argument(
        "--path_to_annotations",
        type=str,
        help="Path to the folder containing the annotations",
        default="/mnt/nas_3dv/hdd1/datasets/3D_FRONT_FUTURE/holistic_layout_20231113/splits/livingroom_test.lst")

    args = parser.parse_args(argv)

    # real_img_folderpath = args.path_to_real_renderings
    # # real_images_path_lst = [os.path.join(real_img_folderpath, f) for f in os.listdir(real_img_folderpath) if f.endswith(".png")]
    # real_images_path_lst = [
    #     os.path.join(real_img_folderpath, f, 'gt_rendered.png')
    #     for f in os.listdir(real_img_folderpath)
    #     if f.isdigit() and os.path.isdir(os.path.join(real_img_folderpath, f))
    # ]
    dataset_folderpath = '/mnt/nas_3dv/hdd1/datasets/3D_FRONT_FUTURE/holistic_layout_20231113/threed_front_livingroom/'
    test_splits_filepath = args.path_to_annotations
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

    print("Generating temporary a folder with test_real images...")
    path_to_test_real = "/tmp/test_real/"
    if not os.path.exists(path_to_test_real):
        os.makedirs(path_to_test_real)
    # for i, img_path in enumerate(real_images_path_lst):
    #     shutil.copyfile(img_path, "{}/{:05d}.png".format(path_to_test_real, i))

    # Number of images to be copied
    N = len(real_images_path_lst)
    print("Number of real images: {}".format(len(real_images_path_lst)))

    fake_img_folderpath = args.path_to_synthesized_renderings
    # fake_images_path_lst = [ os.path.join(fake_img_folderpath, f) for f in os.listdir(fake_img_folderpath) if f.endswith(".png")]
    fake_images_path_lst = [
        os.path.join(fake_img_folderpath, f, 'rendered.png')
        for f in os.listdir(fake_img_folderpath)
        if f.isdigit() and os.path.isdir(os.path.join(fake_img_folderpath, f))
    ]
    print("Generating temporary a folder with test_fake images...")
    path_to_test_fake = "/tmp/test_fake/"
    if not os.path.exists(path_to_test_fake):
        os.makedirs(path_to_test_fake)
    # for i, img_path in enumerate(fake_images_path_lst):
    #     shutil.copyfile(img_path, "{}/{:05d}.png".format(path_to_test_fake, i))
    print("Number of fake images: {}".format(len(fake_images_path_lst)))
    # N = len(fake_images_path_lst)

    scores = []
    for _ in range(10):
        if len(fake_images_path_lst) > N:
            np.random.shuffle(fake_images_path_lst)
            synthesized_images_subset = np.random.choice(fake_images_path_lst, N)
            for i, fi in enumerate(synthesized_images_subset):
                shutil.copyfile(fi, "{}/{:05d}.png".format(path_to_test_fake, i))
            for i, img_path in enumerate(real_images_path_lst):
                shutil.copyfile(img_path, "{}/{:05d}.png".format(path_to_test_real, i))
        else:
            for i, fi in enumerate(fake_images_path_lst):
                shutil.copyfile(fi, "{}/{:05d}.png".format(path_to_test_fake, i))
            selected_real_images = np.random.choice(real_images_path_lst, len(fake_images_path_lst))
            for i, img_path in enumerate(selected_real_images):
                shutil.copyfile(img_path, "{}/{:05d}.png".format(path_to_test_real, i))

        # Compute the FID score
        fid_score = fid.compute_fid(path_to_test_real, path_to_test_fake, device=torch.device("cpu"))
        scores.append(fid_score)
        print(fid_score)
    print(sum(scores) / len(scores))
    print(np.std(scores))


if __name__ == "__main__":
    main(None)
