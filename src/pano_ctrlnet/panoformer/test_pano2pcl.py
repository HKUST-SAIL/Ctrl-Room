from __future__ import absolute_import, division, print_function
import os
import os.path as osp

import numpy as np
import time
import json
import tqdm
import cv2
import open3d as o3d
from PIL import Image, ImageOps

import torch
from torchvision import transforms

torch.manual_seed(100)
torch.cuda.manual_seed(100)

from network.model import Panoformer as PanoBiT

to_tensor_fn = transforms.ToTensor()
normalize_fn = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def save_color_pointcloud_from_panorama(rgb_img_filepath: str, depth_img_filepath: str, saved_color_pcl_filepath: str):

    def get_unit_spherical_map():
        h = 512
        w = 1024
        Theta = np.arange(h).reshape(h, 1) * np.pi / h + np.pi / h / 2
        Theta = np.repeat(Theta, w, axis=1)
        Phi = np.arange(w).reshape(1, w) * 2 * np.pi / w + np.pi / w - np.pi
        # do not flip horizontal
        Phi = np.repeat(Phi, h, axis=0)

        X = np.expand_dims(np.sin(Theta) * np.sin(Phi), 2)
        Y = np.expand_dims(np.cos(Theta), 2)
        Z = np.expand_dims(np.sin(Theta) * np.cos(Phi), 2)
        unit_map = np.concatenate([X, Z, Y], axis=2)

        return unit_map

    def display_inlier_outlier(cloud, ind):
        inlier_cloud = cloud.select_by_index(ind)
        outlier_cloud = cloud.select_by_index(ind, invert=True)

        print("Showing outliers (red) and inliers (gray): ")
        outlier_cloud.paint_uniform_color([1, 0, 0])
        inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
        o3d.visualization.draw([inlier_cloud, outlier_cloud])

    assert osp.exists(rgb_img_filepath), 'rgb panorama doesnt exist!!!'
    assert osp.exists(depth_img_filepath), 'depth panorama doesnt exist!!!'

    raw_depth_img = Image.open(depth_img_filepath)
    if len(raw_depth_img.split()) == 3:
        raw_depth_img = ImageOps.grayscale(raw_depth_img)
    depth_img = np.asarray(raw_depth_img)
    if np.isnan(depth_img.any()) or len(depth_img[depth_img > 0]) == 0:
        print('empyt depth image')
        exit(-1)

    raw_rgb_img = Image.open(rgb_img_filepath)
    rgb_img = np.asarray(raw_rgb_img)
    if np.isnan(rgb_img.any()) or len(rgb_img[rgb_img > 0]) == 0:
        print('empyt rgb image')
        exit(-1)
    color = np.clip(rgb_img, 0.0, 255.0) / 255.0

    depth_img = np.expand_dims((depth_img / 1000.0), axis=2)
    pointcloud = depth_img * get_unit_spherical_map()

    o3d_pointcloud = o3d.geometry.PointCloud()
    o3d_pointcloud.points = o3d.utility.Vector3dVector(pointcloud.reshape(-1, 3))
    o3d_pointcloud.colors = o3d.utility.Vector3dVector(color.reshape(-1, 3))
    # remove outliers
    _, inlier_ind = o3d_pointcloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    # display_inlier_outlier(o3d_pointcloud, ind)
    filtered_o3d_pointcloud = o3d_pointcloud.select_by_index(inlier_ind)
    filtered_o3d_pointcloud.estimate_normals()
    filtered_o3d_pointcloud.orient_normals_consistent_tangent_plane(100)
    o3d.io.write_point_cloud(saved_color_pcl_filepath, filtered_o3d_pointcloud)
    return filtered_o3d_pointcloud


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    load_weights_dir = 'ckpts/mp3d_weights'
    # load_weights_dir = 'ckpts/s2d3d_weights'
    # load panoformer model
    panoformer_model = PanoBiT()
    panoformer_model.to(device)
    assert os.path.isdir(load_weights_dir), "Cannot find folder {}".format(load_weights_dir)
    print("loading model from folder {}".format(load_weights_dir))

    model_path = os.path.join(load_weights_dir, "{}.pth".format("model"))
    model_dict = panoformer_model.state_dict()
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    panoformer_model.load_state_dict(model_dict)

    input_folderpath = '/mnt/nas_3dv/hdd1/datasets/fangchuan/codes/Layout_Controlnet/sample_results/openai-2023-08-17-11-55-58-694085/'
    if not osp.exists(input_folderpath):
        print(f'input_folderpath: {input_folderpath} doesnt exist!!!')
        exit(-1)

    folder_lst = [f for f in os.listdir(input_folderpath) if osp.isdir(osp.join(input_folderpath, f))]
    folder_lst = sorted(folder_lst, key=lambda x: int(x))
    for sub_folder in folder_lst:
        sub_folderpath = osp.join(input_folderpath, sub_folder)
        if not osp.isdir(sub_folderpath):
            continue

        print(f'processing {sub_folderpath}')

        rgb_img_filepath = osp.join(sub_folderpath, '0.png')
        save_depth_img_filepath = osp.join(sub_folderpath, '0_depth.png')

        if osp.exists(save_depth_img_filepath):
            continue

        rgb = cv2.imread(rgb_img_filepath)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        # rgb = cv2.resize(rgb, dsize=(1024, 512), interpolation=cv2.INTER_CUBIC)
        assert rgb.shape == (512, 1024, 3), 'rgb shape is not 512x1024x3!!!'

        input_rgb = to_tensor_fn(rgb.copy()).float()
        inputs = {}
        inputs["rgb"] = input_rgb.unsqueeze(0).to(device)
        inputs["normalized_rgb"] = normalize_fn(input_rgb).unsqueeze(0).to(device)
        print(f'inputs["normalized_rgb"].shape: {inputs["normalized_rgb"].shape}')

        # test on my pano image
        outputs = panoformer_model(inputs["normalized_rgb"])
        pred_depth = outputs["pred_depth"]
        print(f'pred_depth.shape: {pred_depth.shape}')

        # save pred_depth
        pred_depth = pred_depth.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        # print(f'pred_depth: \n {pred_depth}')
        pred_depth = pred_depth * 1000
        pred_depth = pred_depth.astype(np.uint16)
        cv2.imwrite(save_depth_img_filepath, pred_depth)

        # save color point cloud
        saved_color_pcl_filepath = osp.join(sub_folderpath, 'color_pcl.ply')
        color_pcl = save_color_pointcloud_from_panorama(rgb_img_filepath, save_depth_img_filepath,
                                                        saved_color_pcl_filepath)

        # save mesh
        print('run Poisson surface reconstruction')
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(color_pcl, depth=12)
            saved_mesh_filepath = osp.join(sub_folderpath, 'mesh.ply')
            o3d.io.write_triangle_mesh(saved_mesh_filepath, mesh)
