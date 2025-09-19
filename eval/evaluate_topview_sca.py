#
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
#
"""Script used to evaluate the scene classification accuracy between real and
synthesized scenes.
"""
import os
import sys

sys.path.append('.')
sys.path.append('..')
import argparse

from PIL import Image

import numpy as np
import torch
from torchvision import models
import shutil


class ImageFolderDataset(torch.utils.data.Dataset):

    def __init__(self, directory, train=True):
        images = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith("png")])
        N = len(images) // 2

        start = 0 if train else N
        self.images = images[start:start + N]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]


class ThreedFrontRenderDataset(torch.utils.data.Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx].image_path


class SyntheticVRealDataset(torch.utils.data.Dataset):

    def __init__(self, real, synthetic):
        self.N = min(len(real), len(synthetic))
        self.real = real
        self.synthetic = synthetic

    def __len__(self):
        return 2 * self.N

    def __getitem__(self, idx):
        if idx < self.N:
            image_path = self.real[idx]
            label = 1
        else:
            image_path = self.synthetic[idx - self.N]
            label = 0

        img = Image.open(image_path)
        img = np.asarray(img).astype(np.float32) / np.float32(255)
        img = np.transpose(img[:, :, :3], (2, 0, 1))

        return torch.from_numpy(img), torch.tensor([label], dtype=torch.float)


class AlexNet(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.model = models.alexnet(pretrained=True)
        self.fc = torch.nn.Linear(9216, 1)

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        x = self.fc(x.view(len(x), -1))
        x = torch.sigmoid(x)

        return x


class AverageMeter:

    def __init__(self):
        self._value = 0
        self._cnt = 0

    def __iadd__(self, x):
        if torch.is_tensor(x):
            self._value += x.sum().item()
            self._cnt += x.numel()
        else:
            self._value += x
            self._cnt += 1
        return self

    @property
    def value(self):
        return self._value / self._cnt


def mv_images_to_folder(splits_filepath, dataset_root_folderpath, output_folderpath):
    splits = []
    with open(splits_filepath, 'r') as f:
        for line in f:
            splits.append(line.strip())
    real_images_path_lst = [
        os.path.join(dataset_root_folderpath, f, 'rendered_scene_notexture_256.png')
        for f in os.listdir(dataset_root_folderpath)
        if f.split('_')[-1] in splits
    ]
    if not os.path.exists(output_folderpath):
        os.makedirs(output_folderpath)
    for i, img_path in enumerate(real_images_path_lst):
        shutil.copyfile(img_path, os.path.join(output_folderpath, f'{i:05d}.png'))

    return output_folderpath


def main(argv):
    parser = argparse.ArgumentParser(description=("Train a classifier to discriminate between real "
                                                  "and synthetic rooms"))
    # parser.add_argument("--path_to_real_train_renderings",
    #                     type=str,
    #                     help="Path to the folder containing the real renderings of train split",
    #                     default="/mnt/nas_3dv/hdd1/datasets/3D_FRONT_FUTURE/train/bedroom/topview/")
    # parser.add_argument("--path_to_real_test_renderings",
    #                     type=str,
    #                     help="Path to the folder containing the real renderings of test split",
    #                     default="/mnt/nas_3dv/hdd1/datasets/3D_FRONT_FUTURE/test/bedroom/topview/")
    parser.add_argument(
        "--path_to_dataset",
        type=str,
        help="Path to the folder of dataset",
        default='/mnt/nas_3dv/hdd1/datasets/3D_FRONT_FUTURE/holistic_layout_20231113/threed_front_livingroom/')
    # parser.add_argument(
    #     "--path_to_synthesized_renderings",
    #     type=str,
    #     help="Path to the folder containing the synthesized",
    #     default=
    #     "/mnt/nas_3dv/hdd1/fangchuan/HolisticDiffuScene/sample_results/openai-2023-09-23-12-23-59-306819/topview/")
    parser.add_argument(
        "--path_to_synthesized_renderings",
        type=str,
        help="Path to the folder containing the synthesized",
        default=
        "/mnt/nas_3dv/hdd1/fangchuan/HolisticDiffuScene/log/3dfront_diningroom/text-cond-train-test/openai-2023-11-17-16-26-10-740237/diningroom/"
    )
    parser.add_argument(
        "--path_to_test_annotations",
        type=str,
        help="Path to the folder containing the annotations",
        default="/mnt/nas_3dv/hdd1/datasets/3D_FRONT_FUTURE/holistic_layout_20231113/splits/livingroom_test.lst")
    parser.add_argument(
        "--path_to_train_annotations",
        type=str,
        help="Path to the folder containing the annotations",
        default="/mnt/nas_3dv/hdd1/datasets/3D_FRONT_FUTURE/holistic_layout_20231113/splits/livingroom_train.lst")
    parser.add_argument("--batch_size",
                        type=int,
                        default=256,
                        help="Set the batch size for training and evaluating (default: 256)")
    parser.add_argument("--num_workers", type=int, default=0, help="Set the PyTorch data loader workers (default: 0)")
    parser.add_argument("--epochs", type=int, default=10, help="Train for that many epochs (default: 10)")
    parser.add_argument("--output_directory", default="/mnt/nas_3dv/hdd1/fangchuan/eccv2024_other_methods/diffuscene_exps/test_livingroom_sca/", help="Path to the output directory")
    args = parser.parse_args(argv)

    if torch.cuda.is_available():
        device = torch.device("cuda:1")
    else:
        device = torch.device("cpu")
    print("Running code on", device)

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    dataset_folderpath = args.path_to_dataset
    train_splits_filepath = args.path_to_train_annotations
    test_splits_filepath = args.path_to_test_annotations

    real_train_renderings_folder = os.path.join(args.output_directory, 'real_train_renderings')
    mv_images_to_folder(train_splits_filepath, dataset_folderpath, real_train_renderings_folder)
    real_train_img_num = len([f for f in os.listdir(real_train_renderings_folder) if f.endswith('.png')])
    print(f'move {real_train_img_num} images to {real_train_renderings_folder}')

    real_test_renderings_folder = os.path.join(args.output_directory, 'real_test_renderings')
    mv_images_to_folder(test_splits_filepath, dataset_folderpath, real_test_renderings_folder)
    real_test_img_num = len([f for f in os.listdir(real_test_renderings_folder) if f.endswith('.png')])
    print(f'move {real_test_img_num} images to {real_test_renderings_folder}')

    # Create Real datasets
    train_real = ImageFolderDataset(real_train_renderings_folder, train=True)
    test_real = ImageFolderDataset(real_test_renderings_folder, train=False)

    # Create the synthetic datasets
    # fake_img_folderpath = args.path_to_synthesized_renderings
    # fake_images_path_lst = [
    #     os.path.join(fake_img_folderpath, f, 'rendered.png')
    #     for f in os.listdir(fake_img_folderpath)
    #     if f.isdigit() and os.path.isdir(os.path.join(fake_img_folderpath, f))
    # ]
    # path_to_test_fake = "/tmp/test_fake/"
    # if not os.path.exists(path_to_test_fake):
    #     os.makedirs(path_to_test_fake)
    # for i, sythesized_img_path in enumerate(fake_images_path_lst):
    #     shutil.copyfile(sythesized_img_path, os.path.join(path_to_test_fake, f'{i:05d}.png'))
    
    # path_to_test_fake = '/mnt/nas_3dv/hdd1/fangchuan/eccv2024_other_methods/diffuscene_exps/test_diningroom_fake'
    path_to_test_fake = '/mnt/nas_3dv/hdd1/fangchuan/eccv2024_other_methods/diffuscene_exps/test_livingroom_fake/'
    train_synthetic = ImageFolderDataset(directory=path_to_test_fake, train=True)
    test_synthetic = ImageFolderDataset(directory=path_to_test_fake, train=False)

    # Join them in useable datasets
    train_dataset = SyntheticVRealDataset(train_real, train_synthetic)
    test_dataset = SyntheticVRealDataset(test_real, test_synthetic)

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=args.num_workers)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  num_workers=args.num_workers)

    # Create the model
    model = AlexNet()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Train the model
    scores = []
    for _ in range(10):
        for e in range(args.epochs):
            loss_meter = AverageMeter()
            acc_meter = AverageMeter()
            for i, (x, y) in enumerate(train_dataloader):
                model.train()
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                y_hat = model(x)
                loss = torch.nn.functional.binary_cross_entropy(y_hat, y)
                acc = (torch.abs(y - y_hat) < 0.5).float().mean()
                loss.backward()
                optimizer.step()

                loss_meter += loss
                acc_meter += acc

                msg = "{: 3d} loss: {:.4f} - acc: {:.4f}".format(i, loss_meter.value, acc_meter.value)
                print(msg + "\b" * len(msg), end="", flush=True)
            print()

            if (e + 1) % 5 == 0:
                with torch.no_grad():
                    model.eval()
                    loss_meter = AverageMeter()
                    acc_meter = AverageMeter()
                    for i, (x, y) in enumerate(test_dataloader):
                        x = x.to(device)
                        y = y.to(device)
                        y_hat = model(x)
                        loss = torch.nn.functional.binary_cross_entropy(y_hat, y)
                        acc = (torch.abs(y - y_hat) < 0.5).float().mean()

                        loss_meter += loss
                        acc_meter += acc

                        msg_pre = "{: 3d} val_loss: {:.4f} - val_acc: {:.4f}"

                        msg = msg_pre.format(i, loss_meter.value, acc_meter.value)
                        print(msg + "\b" * len(msg), end="", flush=True)
                    print()
        scores.append(acc_meter.value)
    print(sum(scores) / len(scores))
    print(np.std(scores))


if __name__ == "__main__":
    main(None)
