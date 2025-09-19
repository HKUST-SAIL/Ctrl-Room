import os
import sys

import shutil
import numpy as np
import argparse

import torch
from cleanfid import fid
from PIL import Image
from torchvision import models


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

def compute_sca(path_to_dataset:str, 
                dataset_train_splits_filepath:str, 
                dataset_test_splits_filepath:str, 
                path_to_synthesized_scenes:int, 
                output_directory:str, 
                batch_size:int=256, 
                num_workers:int=0, 
                epochs:int=10, 
                device=torch.device("cpu")):
    """ compute scene classification accuracy (SCA) between the real and the synthetic scene layout

    Args:
        path_to_dataset (str): livingroom/diningroom dataset folder
        dataset_train_splits_filepath (str): train split file path
        dataset_test_splits_filepath (str): test split file path
        path_to_synthesized_scenes (int): synthesized scenes folder, all the rendered images are in this folder
        output_directory (str): output folder for sca calculation
        batch_size (int, optional): _description_. Defaults to 256.
        num_workers (int, optional): _description_. Defaults to 0.
        epochs (int, optional): _description_. Defaults to 10.
        device (_type_, optional): _description_. Defaults to torch.device("cpu").
    """
    real_train_renderings_folder = os.path.join(output_directory, 'real_train_renderings')
    mv_images_to_folder(dataset_train_splits_filepath, path_to_dataset, real_train_renderings_folder)
    real_train_img_num = len([f for f in os.listdir(real_train_renderings_folder) if f.endswith('.png')])
    print(f'move {real_train_img_num} images to {real_train_renderings_folder}')

    real_test_renderings_folder = os.path.join(output_directory, 'real_test_renderings')
    mv_images_to_folder(dataset_test_splits_filepath, dataset_folderpath, real_test_renderings_folder)
    real_test_img_num = len([f for f in os.listdir(real_test_renderings_folder) if f.endswith('.png')])
    print(f'move {real_test_img_num} images to {real_test_renderings_folder}')

    # Create Real datasets
    train_real = ImageFolderDataset(real_train_renderings_folder, train=True)
    test_real = ImageFolderDataset(real_test_renderings_folder, train=False)

    # Create the synthetic datasets
    train_synthetic = ImageFolderDataset(directory=path_to_synthesized_scenes, train=True)
    test_synthetic = ImageFolderDataset(directory=path_to_synthesized_scenes, train=False)

    # Join them in useable datasets
    train_dataset = SyntheticVRealDataset(train_real, train_synthetic)
    test_dataset = SyntheticVRealDataset(test_real, test_synthetic)

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=num_workers)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=num_workers)

    # Create the model
    model = AlexNet()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Train the model
    scores = []
    for _ in range(10):
        for e in range(epochs):
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
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=("Compute the FID/KID/SCA scores between the real and the "
                                                  "synthetic scene layout"))
    parser.add_argument(
        "--path_to_dataset",
        type=str,
        help="Path to the folder containing the real renderings",
        default="/mnt/nas_3dv/hdd1/datasets/3D_FRONT_FUTURE/holistic_layout_20231113/livingroom_topdown_renders/"
    )
    parser.add_argument(
        "--path_to_synthesized_scenes",
        type=str,
        help="Path to the folder containing the synthesized",
        default=
        "/mnt/nas_3dv/hdd1/fangchuan/HolisticDiffuScene/log/3dfront_livingroom/textconditional-0122/sample_results/openai-2024-01-27-08-16-04-293807/livingroom"
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
    
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda:1")
    else:
        device = torch.device("cpu")
    print("Running code on", device)

    dataset_folderpath = args.path_to_dataset
    train_splits_filepath = args.path_to_train_annotations
    test_splits_filepath = args.path_to_test_annotations
    output_folder = args.output_directory
    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Compute the FID and KID score
    path_to_all_real_imgs = '/mnt/nas_3dv/hdd1/fangchuan/eccv2024_other_methods/diffuscene_exps/test_diningroom_real/'
    path_to_all_fake_imgs = '/mnt/nas_3dv/hdd1/fangchuan/eccv2024_other_methods/diffuscene_exps/test_diningroom_fake/'
    # cp renderings to tmp folder
    os.makedirs(path_to_all_real_imgs, exist_ok=True)
    path_to_all_real_renderings = [os.path.join(dataset_folderpath, f, 'rendered_scene_notexture_256.png') for f in os.listdir(dataset_folderpath) if os.path.isdir(os.path.join(dataset_folderpath, f))]
    for i, img_path in enumerate(path_to_all_real_renderings):
        shutil.copyfile(img_path, os.path.join(path_to_all_real_imgs, f'{i:05d}.png'))
    
    os.makedirs(path_to_all_fake_imgs, exist_ok=True)
    path_to_all_fake_renderings = [os.path.join(args.path_to_synthesized_scenes, f) for f in os.listdir(args.path_to_synthesized_scenes) if f.endswith('.png')]
    for i, img_path in enumerate(path_to_all_fake_renderings):
        shutil.copyfile(img_path, os.path.join(path_to_all_fake_imgs, f'{i:05d}.png'))
        
    fid_score = fid.compute_fid(path_to_all_real_imgs, path_to_all_fake_imgs, device=torch.device("cpu"))
    print('fid score:', fid_score)
    kid_score = fid.compute_kid(path_to_all_real_imgs, path_to_all_fake_imgs, device=torch.device("cpu"))
    print('kid score:', kid_score)

    # Compute the SCA score
    compute_sca(path_to_dataset=dataset_folderpath, 
                dataset_train_splits_filepath=train_splits_filepath, 
                dataset_test_splits_filepath=test_splits_filepath, 
                path_to_synthesized_scenes=path_to_all_fake_imgs, 
                output_directory=output_folder, 
                batch_size=args.batch_size, 
                num_workers=args.num_workers, 
                epochs=args.epochs, 
                device=device)

    