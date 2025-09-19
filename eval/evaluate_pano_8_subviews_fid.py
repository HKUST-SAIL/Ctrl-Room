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
    parser.add_argument("--path_to_real_renderings",
                        type=str,
                        help="Path to the folder containing the real renderings",
                        default="/mnt/nas_3dv/hdd1/datasets/3D_FRONT_FUTURE/test/bedroom/topview/")
    parser.add_argument(
        "--path_to_synthesized_renderings",
        type=str,
        help="Path to the folder containing the synthesized",
        default=
        "/mnt/nas_3dv/hdd1/fangchuan/HolisticDiffuScene/sample_results/openai-2023-09-23-12-23-59-306819/topview/")
    parser.add_argument("--path_to_annotations",
                        type=str,
                        help="Path to the folder containing the annotations",
                        default="/data/3dfuture/3dfront/annotations/")

    args = parser.parse_args(argv)

    real_img_folderpath = args.path_to_real_renderings
    real_images_path_lst = [
        os.path.join(real_img_folderpath, f) for f in os.listdir(real_img_folderpath) if f.endswith(".png")
    ]
    print("Generating temporary a folder with test_real images...")
    path_to_test_real = "/tmp/test_real/"
    if not os.path.exists(path_to_test_real):
        os.makedirs(path_to_test_real)
    for i, img_path in enumerate(real_images_path_lst):
        shutil.copyfile(img_path, "{}/{:05d}.png".format(path_to_test_real, i))

    # Number of images to be copied
    N = len(real_images_path_lst)
    print("Number of real images: {}".format(len(real_images_path_lst)))

    fake_img_folderpath = args.path_to_synthesized_renderings
    fake_images_path_lst = [
        os.path.join(fake_img_folderpath, f) for f in os.listdir(fake_img_folderpath) if f.endswith(".png")
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
        np.random.shuffle(fake_images_path_lst)
        synthesized_images_subset = np.random.choice(fake_images_path_lst, N)
        for i, fi in enumerate(synthesized_images_subset):
            shutil.copyfile(fi, "{}/{:05d}.png".format(path_to_test_fake, i))
        # np.random.shuffle(real_images_path_lst)
        # real_images_subset = np.random.choice(real_images_path_lst, N)
        # for i, fi in enumerate(real_images_subset):
        #     shutil.copyfile(fi, "{}/{:05d}.png".format(path_to_test_real, i))

        # Compute the FID score
        fid_score = fid.compute_fid(path_to_test_real, path_to_test_fake, device=torch.device("cpu"))
        scores.append(fid_score)
        print(fid_score)
    print(sum(scores) / len(scores))
    print(np.std(scores))


if __name__ == "__main__":
    main(None)
