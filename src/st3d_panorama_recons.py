import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import sys

sys.path.append(".")
sys.path.append("..")
import argparse
import gc
import glob
import re
from typing import Tuple, Union
import shutil

import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2
import open3d as o3d

from src.pano_ctrlnet.panoformer.network.model import Panoformer as PanoBiT
from src.pano_ctrlnet.egformer.egformer import EGDepthModel


def get_rank():
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0


def get_device():
    return torch.device(f"cuda:{get_rank()}")


def load_module_weights(path, module_name=None, ignore_modules=None, map_location=None) -> Tuple[dict, int, int]:
    if module_name is not None and ignore_modules is not None:
        raise ValueError("module_name and ignore_modules cannot be both set")
    if map_location is None:
        map_location = get_device()

    ckpt = torch.load(path, map_location=map_location)
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    state_dict_to_load = state_dict

    if ignore_modules is not None:
        state_dict_to_load = {}
        for k, v in state_dict.items():
            ignore = any([k.startswith(ignore_module + ".") for ignore_module in ignore_modules])
            if ignore:
                # print(f'ignore k: {k}')
                continue
            state_dict_to_load[k] = v

    if module_name is not None:
        state_dict_to_load = {}
        for k, v in state_dict.items():
            m = re.match(rf"^{module_name}\.(.*)$", k)
            if m is None:
                continue
            state_dict_to_load[k] = v

    return state_dict_to_load


# load panoformer model
def load_depth_estimator_model(device: str = "cuda"):
    load_weights_dir = "../ckpts/panoformer_mp3d_weights"
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
    panoformer_model.eval()

    return panoformer_model


# load egformer model
def load_depth_estimator_egformer_model(device: str = "cuda"):
    model_path = "ckpts/EGformer_pretrained.pkl"
    print("loading model from folder {}".format(model_path))

    egformer_model = EGDepthModel(hybrid=False)
    state_dict = load_module_weights(path=model_path, map_location="cpu")
    egformer_model.load_state_dict(state_dict, strict=False)

    egformer_model.to(device)
    egformer_model.eval()

    return egformer_model


def gen_depth_mesh(
    rgb: np.ndarray,
    device: str,
    depth_estimation_model: Union[PanoBiT, EGDepthModel],
    poisson_exe_path: str = "./PoissonRecon",
    output_folderpath: str = None,
):
    def gen_pointcloud_from_panorama(raw_rgb_img: np.ndarray, raw_depth_img: np.ndarray, depth_scale: float = 1000.0):

        # def get_unit_spherical_map():
        #     h = 512
        #     w = 1024
        #     Theta = np.arange(h).reshape(h, 1) * np.pi / h + np.pi / h / 2
        #     Theta = np.repeat(Theta, w, axis=1)
        #     Phi = np.arange(w).reshape(1, w) * 2 * np.pi / w + np.pi / w - np.pi
        #     # do not flip horizontal
        #     Phi = np.repeat(Phi, h, axis=0)

        #     X = np.expand_dims(np.sin(Theta) * np.sin(Phi), 2)
        #     Y = np.expand_dims(np.cos(Theta), 2)
        #     Z = np.expand_dims(np.sin(Theta) * np.cos(Phi), 2)
        #     unit_map = np.concatenate([X, Z, Y], axis=2)

        #     return unit_map
        def np_coorx2u(coorx, coorW=1024):
            return ((coorx + 0.5) / coorW - 0.5) * 2 * np.pi

        def np_coory2v(coory, coorH=512):
            return -((coory + 0.5) / coorH - 0.5) * np.pi

        def get_unit_spherical_map(h=512, w=1024):
            coorx, coory = np.meshgrid(np.arange(w), np.arange(h))
            us = np_coorx2u(coorx, w)
            vs = np_coory2v(coory, h)

            X = np.expand_dims(np.cos(vs) * np.sin(us), 2)
            Y = np.expand_dims(np.sin(vs), 2)
            Z = np.expand_dims(np.cos(vs) * np.cos(us), 2)
            unit_map = np.concatenate([X, Z, Y], axis=2)
            return unit_map

        depth_img = np.asarray(raw_depth_img)
        if np.isnan(depth_img.any()) or len(depth_img[depth_img > 0]) == 0:
            print("empyt depth image")
            exit(-1)

        if len(depth_img.shape) == 3:
            depth_img = depth_img[:, :, 0]
        depth_img = np.expand_dims((depth_img / depth_scale), axis=2)
        pointcloud = depth_img * get_unit_spherical_map()

        o3d_pointcloud = o3d.geometry.PointCloud()
        o3d_pointcloud.points = o3d.utility.Vector3dVector(pointcloud.reshape(-1, 3))
        return o3d_pointcloud

    to_tensor_fn = transforms.ToTensor()
    normalize_fn = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    assert rgb.shape == (512, 1024, 3), "rgb shape is not 512x1024x3!!!"

    saved_depth_img_filepath = os.path.join(output_folderpath, "depth.png")

    if os.path.exists(saved_depth_img_filepath):
        print(f"load finetuned depth image from {saved_depth_img_filepath}")
        pred_depth = np.array(Image.open(saved_depth_img_filepath))
    else:
        input_rgb = to_tensor_fn(rgb.copy()).float()
        inputs = {}
        inputs["rgb"] = input_rgb.unsqueeze(0).to(device)
        inputs["normalized_rgb"] = normalize_fn(input_rgb).unsqueeze(0).to(device)

        # test on my pano image
        if isinstance(depth_estimation_model, PanoBiT):
            outputs = depth_estimation_model(inputs["normalized_rgb"])
            pred_depth = outputs["pred_depth"]
            pred_depth = pred_depth.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

        elif isinstance(depth_estimation_model, EGDepthModel):
            outputs = depth_estimation_model(inputs["rgb"])
            pred_depth = outputs.detach().cpu().numpy().squeeze()
            # scale up the predicted depth
            pred_depth *= 10.0

        # save pred_depth
        pred_depth = pred_depth * 1000
        pred_depth = pred_depth.astype(np.uint16)
        cv2.imwrite(saved_depth_img_filepath, pred_depth)

    # save point cloud
    print(f"run point cloud generation...")
    saved_pcl_filepath = os.path.join(output_folderpath, "pointcloud.ply")
    pcl = gen_pointcloud_from_panorama(rgb, pred_depth, depth_scale=1000.0)

    # downsample point cloud
    pcl = pcl.voxel_down_sample(voxel_size=0.03)
    # rotate point cloud
    R_x_90 = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    pcl.rotate(R_x_90, center=(0, 0, 0))
    pcl.translate((0, 1.6, 0))
    # must constrain normals pointing towards camera
    pcl.estimate_normals()
    pcl.orient_normals_towards_camera_location(camera_location=(0, 1.6, 0))
    o3d.io.write_point_cloud(saved_pcl_filepath, pcl)

    # /home/fangchuan/codes/libs/PoissonRecon/Bin/Linux/PoissonRecon
    # --in /mnt/nas_3dv/hdd1/fangchuan/HolisticDiffuScene/sample_results/openai-2023-09-09-17-51-37-694284/livingroom/0/pointcloud.ply
    # --out /mnt/nas_3dv/hdd1/fangchuan/HolisticDiffuScene/sample_results/openai-2023-09-09-17-51-37-694284/livingroom/0/poisson_mesh.ply
    # --depth 10 --ascii --verbose --threads 1
    # save mesh
    print("run Poisson surface reconstruction...")
    saved_mesh_filepath = os.path.join(output_folderpath, "mesh.ply")
    cmd = f"{poisson_exe_path} --in {saved_pcl_filepath} --out {saved_mesh_filepath} --depth 10 --ascii --verbose --threads 1"
    os.system(cmd)
    # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    #     mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcl, depth=10)
    #     o3d.io.write_triangle_mesh(saved_mesh_filepath, mesh)

    return saved_pcl_filepath, saved_mesh_filepath


def main(args):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load control images
    input_folders_lst = [
        os.path.join(args.input_folder, f)
        for f in os.listdir(args.input_folder)
        if os.path.isdir(os.path.join(args.input_folder, f))
    ]

    gc.collect()
    torch.cuda.empty_cache()

    is_blockade_pano = False
    is_text2light_pano = False
    is_mvdiffusion_pano = False

    # load depth estimation model
    if args.use_egformer:
        depth_estimation_model = load_depth_estimator_egformer_model(device)
    else:
        depth_estimation_model = load_depth_estimator_model(device)

    def resize_panorama(img_folder: str):
        img_path = glob.glob(os.path.join(img_folder, "*.png"))
        if len(img_path) == 1:
            img_path = img_path[0]
        elif len(img_path) == 2:
            img_path = img_path[1] if "_pano.png" in os.path.basename(img_path[0]) else img_path[0]
        img_name = img_path.split("/")[-1]
        print(f"Resize panorama  {img_name}")
        img = Image.open(img_path).convert("RGB")
        pano_img = img.resize((1024, 512))
        img_path = os.path.join(img_folder, img_name[:-4] + "_pano.png")
        pano_img.save(img_path)
        print(f"Resize sr panorama to {img_path}")
        sr_pano_img = img.resize((2048, 1024))
        img_path = os.path.join(img_folder, img_name[:-4] + "_pano_sr.png")
        sr_pano_img.save(img_path)

    def resize_openexr_panorama(img_folder: str):
        img_path = glob.glob(os.path.join(img_folder, "*_hdr.png"))
        img_path = img_path[0]
        img_name = img_path.split("/")[-1]
        print(f"Resize panorama  {img_name}")
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        # convert uint16 to uint8
        img = cv2.convertScaleAbs(img, alpha=(255.0 / 65535.0))
        img_path = os.path.join(img_folder, img_name[:-4] + "_pano.png")
        pano_img = cv2.resize(img, (1024, 512), interpolation=cv2.INTER_AREA)
        cv2.imwrite(img_path, pano_img[:, :, :3])
        print(f"Resize sr panorama to {img_path}")
        # sr_pano_img = cv2.resize(normalized_img, (2048,1024), interpolation=cv2.INTER_AREA)
        img_path = os.path.join(img_folder, img_name[:-4] + "_pano_sr.png")
        sr_pano_img = cv2.resize(img, (2048, 1024), interpolation=cv2.INTER_AREA)
        cv2.imwrite(img_path, sr_pano_img[:, :, :3])

    def rename_panorama(img_folder: str):
        pano_img_path = glob.glob(os.path.join(img_folder, "pano.png"))
        if len(pano_img_path) == 0:
            print(f"Cannot find pano.png in {img_folder}")
            exit(-1)

        scene_name = os.path.basename(img_folder)
        shutil.copy(pano_img_path[0], os.path.join(img_folder, f"{scene_name}_pano.png"))

        sr_pano_img_path = glob.glob(os.path.join(img_folder, "pano_sr.png"))
        if len(sr_pano_img_path) == 0:
            print(f"Cannot find pano_sr.png in {img_folder}")
            exit(-1)
        shutil.copy(sr_pano_img_path[0], os.path.join(img_folder, f"{scene_name}_pano_sr.png"))

    for i, img_folder in enumerate(input_folders_lst):

        output_folder = img_folder
        if is_blockade_pano:
            resize_panorama(img_folder)
        elif is_text2light_pano:
            resize_openexr_panorama(img_folder)
        elif is_mvdiffusion_pano:
            rename_panorama(img_folder)
        img_path = glob.glob(os.path.join(img_folder, "*_pano.png"))
        if len(img_path) == 0:
            continue
        img_path = img_path[0]
        img_name = img_path.split("/")[-1]
        print(f"Predict depth  {i}/{len(input_folders_lst)}:  {img_name}")

        sampled_img = Image.open(img_path).convert("RGB")
        sampled_img = np.array(sampled_img)
        # predict depth and mesh
        pcl_filepath, poisson_mesh_filepath = gen_depth_mesh(
            sampled_img,
            device,
            depth_estimation_model,
            poisson_exe_path=args.poisson_exe_path,
            output_folderpath=output_folder,
        )

        print("run mesh texturing...")
        saved_textured_mesh_filepath = os.path.join(output_folder, "model.obj")
        sr_img_path_lst = glob.glob(os.path.join(img_folder, "*_pano.png"))
        if len(sr_img_path_lst) == 0:
            sr_img_path = img_path
        else:
            sr_img_path = sr_img_path_lst[0]
        mesh_tex_exe_path = args.mesh_tex_exe_path
        cmd = f"{mesh_tex_exe_path} {sr_img_path} {poisson_mesh_filepath} {output_folder}"
        os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, default="sample_results/livingroom")
    parser.add_argument("--poisson_exe_path", type=str, default="/app/libs/PoissonRecon/Bin/Linux/PoissonRecon")
    parser.add_argument(
        "--mesh_tex_exe_path", type=str, default="/app/libs/PanoTexturing/build/apps/pano_texrecon/panorecons"
    )
    parser.add_argument("--use_egformer", type=bool, default=False)
    args = parser.parse_args()
    main(args)
