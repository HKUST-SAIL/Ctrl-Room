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

import shutil

from torchmetrics.image.kid import KernelInceptionDistance

kid_fn = KernelInceptionDistance(subset_size=50)


def main(argv):
    # parser = argparse.ArgumentParser(description=("Compute the FID scores between the real and the "
    #                                               "synthetic images"))

    # parser.add_argument(
    #     "--path_to_real_renderings",
    #     type=str,
    #     help="Path to the folder containing the real renderings",
    #     default="/mnt/nas_3dv/hdd1/datasets/3D_FRONT_FUTURE/holistic_layout_20231113/livingroom_topdown_renders/")
    # parser.add_argument(
    #     "--path_to_synthesized_renderings",
    #     type=str,
    #     help="Path to the folder containing the synthesized",
    #     default=
    #     # "/mnt/nas_3dv/hdd1/fangchuan/HolisticDiffuScene/log/3dfront_livingroom/unconditional-new/sample_results/openai-2023-12-05-20-07-23-532451/livingroom"
    #     "/mnt/nas_3dv/hdd1/fangchuan/HolisticDiffuScene/log/3dfront_livingroom/textconditional-0122/sample_results/openai-2024-01-27-08-16-04-293807/livingroom"
    # )

    # parser.add_argument(
    #     "--path_to_annotations",
    #     type=str,
    #     help="Path to the folder containing the annotations",
    #     default="/mnt/nas_3dv/hdd1/datasets/3D_FRONT_FUTURE/holistic_layout_20231113/splits/diningroom_test.lst")

    # args = parser.parse_args(argv)

    # # real_img_folderpath = args.path_to_real_renderings
    # # real_images_path_lst = [
    # #     os.path.join(real_img_folderpath, f) for f in os.listdir(real_img_folderpath) if f.endswith(".png")
    # # ]
    # # real_images_path_lst = [
    # #     os.path.join(real_img_folderpath, f, 'gt_rendered.png')
    # #     for f in os.listdir(real_img_folderpath)
    # #     if f.isdigit() and os.path.isdir(os.path.join(real_img_folderpath, f))
    # # ]
    # dataset_folderpath = '/mnt/nas_3dv/hdd1/datasets/3D_FRONT_FUTURE/holistic_layout_20231113/threed_front_diningroom/'
    # test_splits_filepath = args.path_to_annotations
    # train_splits_filepath = test_splits_filepath.replace('test', 'train')
    # valid_splits_filepath = test_splits_filepath.replace('test', 'val')

    # test_splits = []
    # with open(test_splits_filepath, 'r') as f:
    #     for line in f:
    #         test_splits.append(line.strip())
    # with open(train_splits_filepath, 'r') as f:
    #     for line in f:
    #         test_splits.append(line.strip())
    # with open(valid_splits_filepath, 'r') as f:
    #     for line in f:
    #         test_splits.append(line.strip())
    # real_images_path_lst = [
    #     os.path.join(dataset_folderpath, f, 'rendered_scene_notexture_256.png')
    #     for f in os.listdir(dataset_folderpath)
    #     if f.split('_')[-1] in test_splits
    # ]

    # print("Generating temporary a folder with test_real images...")
    # path_to_test_real = "/tmp/test_real/"
    # if not os.path.exists(path_to_test_real):
    #     os.makedirs(path_to_test_real)

    # real_img_lst = []
    # for i, img_path in enumerate(real_images_path_lst):
    #     shutil.copyfile(img_path, "{}/{:05d}.png".format(path_to_test_real, i))
    #     real_img_lst.append(np.array(Image.open(img_path).convert("RGB")))
    # imgs_dist1 = torch.from_numpy(np.array(real_img_lst)).permute(0, 3, 1, 2).to(torch.uint8)

    # # Number of images to be copied
    # N = len(real_images_path_lst)
    # print("Number of real images: {}".format(len(real_images_path_lst)))

    # fake_img_folderpath = args.path_to_synthesized_renderings
    # # fake_images_path_lst = [
    # #     os.path.join(fake_img_folderpath, f) for f in os.listdir(fake_img_folderpath) if f.endswith(".png")
    # # ]
    # fake_images_path_lst = [
    #     os.path.join(fake_img_folderpath, f, 'rendered.png')
    #     for f in os.listdir(fake_img_folderpath)
    #     if f.isdigit() and os.path.isdir(os.path.join(fake_img_folderpath, f))
    # ]
    # print("Generating temporary a folder with test_fake images...")
    # path_to_test_fake = "/tmp/test_fake/"
    # if not os.path.exists(path_to_test_fake):
    #     os.makedirs(path_to_test_fake)
    # # for i, img_path in enumerate(fake_images_path_lst):
    # #     shutil.copyfile(img_path, "{}/{:05d}.png".format(path_to_test_fake, i))
    # print("Number of fake images: {}".format(len(fake_images_path_lst)))
    # # N = len(fake_images_path_lst)
    
    # scores = []
    # for _ in range(10):
    #     np.random.shuffle(fake_images_path_lst)
    #     synthesized_images_subset = np.random.choice(fake_images_path_lst, N)

    #     fake_img_lst = []
    #     for i, fi in enumerate(synthesized_images_subset):
    #         shutil.copyfile(fi, "{}/{:05d}.png".format(path_to_test_fake, i))
    #         fake_img_lst.append(np.array(Image.open(fi).convert("RGB")))
    #     imgs_dist2 = torch.from_numpy(np.array(fake_img_lst)).permute(0, 3, 1, 2).to(torch.uint8)

    #     # Compute the FID score
    #     kid_fn.update(imgs_dist1, real=True)
    #     kid_fn.update(imgs_dist2, real=False)
    #     kid_mean, kid_std = kid_fn.compute()

    #     scores.append(kid_mean)
    #     print(kid_mean)
    # print(sum(scores) / len(scores))
    # print(np.std(scores))

    real_img_folderpath = '/mnt/nas_3dv/hdd1/fangchuan/eccv2024_other_methods/diffuscene_exps/test_diningroom_real/'
    real_img_path_lst = [f for f in os.listdir(real_img_folderpath) if f.endswith(".png")]
    real_img_lst = []
    for i, img_path in enumerate(real_img_path_lst):
        img_path = os.path.join(real_img_folderpath, img_path)
        real_img_lst.append(np.array(Image.open(img_path).convert("RGB")))
    N_real = len(real_img_lst)
    imgs_dist1 = torch.from_numpy(np.array(real_img_lst)).permute(0, 3, 1, 2).to(torch.uint8)
    
    fake_img_folderpath = '/mnt/nas_3dv/hdd1/fangchuan/eccv2024_other_methods/diffuscene_exps/test_fake/'
    fake_img_path_lst = [f for f in os.listdir(fake_img_folderpath) if f.endswith(".png")]
    fake_img_lst = []
    for i, img_path in enumerate(fake_img_path_lst):
        img_path = os.path.join(fake_img_folderpath, img_path)
        fake_img_lst.append(np.array(Image.open(img_path).convert("RGB")))
    N_fake = len(fake_img_lst)
    
    scores = []
    for _ in range(10):
        np.random.shuffle(fake_img_lst)
        synthesized_image_path_subset = np.random.choice(fake_img_path_lst, N_real)
        fake_img_lst = []
        for i, img_path in enumerate(synthesized_image_path_subset):
            img_path = os.path.join(fake_img_folderpath, img_path)
            fake_img_lst.append(np.array(Image.open(img_path).convert("RGB")))
        imgs_dist2 = torch.from_numpy(np.array(fake_img_lst)).permute(0, 3, 1, 2).to(torch.uint8)

        # Compute the FID score
        kid_fn.update(imgs_dist1, real=True)
        kid_fn.update(imgs_dist2, real=False)
        kid_mean, kid_std = kid_fn.compute()

        scores.append(kid_mean)
        print(kid_mean)
    print(sum(scores) / len(scores))
    print(np.std(scores))


if __name__ == "__main__":
    main(None)
