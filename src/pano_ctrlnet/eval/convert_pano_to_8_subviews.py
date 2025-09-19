# this script is used to convert pano images to 8 subviews for MVDiffusion/Text2Room/Text2Light/Ours

import os
import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import glob
import argparse
import cv2
import numpy as np

import torch
from utils import warp_img, Equirectangular
from PIL import Image
from tqdm import tqdm

from scipy.ndimage import map_coordinates


# Based on https://github.com/sunset1995/py360convert
class Equirec2Cube:
    def __init__(self, equ_h:int=512, equ_w:int=1024, face_w:int=512):
        '''
        equ_h: int, height of the equirectangular image
        equ_w: int, width of the equirectangular image
        face_w: int, the length of each face of the cubemap
        '''

        self.equ_h = equ_h
        self.equ_w = equ_w
        self.face_w = face_w

        self._xyzcube()
        self._xyz2coor()

        # For convert R-distance to Z-depth for CubeMaps
        cosmap = 1 / np.sqrt((2 * self.grid[..., 0]) ** 2 + (2 * self.grid[..., 1]) ** 2 + 1)
        self.cosmaps = np.concatenate(6 * [cosmap], axis=1)[..., np.newaxis]

    def _xyzcube(self):
        '''
        Compute the xyz cordinates of the unit cube in [F R B L U D] format.
        '''
        self.xyz = np.zeros((self.face_w, self.face_w * 6, 3), np.float32)
        rng = np.linspace(-0.5, 0.5, num=self.face_w, dtype=np.float32)
        self.grid = np.stack(np.meshgrid(rng, -rng), -1)

        # Front face (z = 0.5)
        self.xyz[:, 0 * self.face_w:1 * self.face_w, [0, 1]] = self.grid
        self.xyz[:, 0 * self.face_w:1 * self.face_w, 2] = 0.5

        # Right face (x = 0.5)
        self.xyz[:, 1 * self.face_w:2 * self.face_w, [2, 1]] = self.grid[:, ::-1]
        self.xyz[:, 1 * self.face_w:2 * self.face_w, 0] = 0.5

        # Back face (z = -0.5)
        self.xyz[:, 2 * self.face_w:3 * self.face_w, [0, 1]] = self.grid[:, ::-1]
        self.xyz[:, 2 * self.face_w:3 * self.face_w, 2] = -0.5

        # Left face (x = -0.5)
        self.xyz[:, 3 * self.face_w:4 * self.face_w, [2, 1]] = self.grid
        self.xyz[:, 3 * self.face_w:4 * self.face_w, 0] = -0.5

        # Up face (y = 0.5)
        self.xyz[:, 4 * self.face_w:5 * self.face_w, [0, 2]] = self.grid[::-1, :]
        self.xyz[:, 4 * self.face_w:5 * self.face_w, 1] = 0.5

        # Down face (y = -0.5)
        self.xyz[:, 5 * self.face_w:6 * self.face_w, [0, 2]] = self.grid
        self.xyz[:, 5 * self.face_w:6 * self.face_w, 1] = -0.5

    def _xyz2coor(self):

        # x, y, z to longitude and latitude
        x, y, z = np.split(self.xyz, 3, axis=-1)
        lon = np.arctan2(x, z)
        c = np.sqrt(x ** 2 + z ** 2)
        lat = np.arctan2(y, c)

        # longitude and latitude to equirectangular coordinate
        self.coor_x = (lon / (2 * np.pi) + 0.5) * self.equ_w - 0.5
        self.coor_y = (-lat / np.pi + 0.5) * self.equ_h - 0.5

    def sample_equirec(self, e_img, order=0):
        pad_u = np.roll(e_img[[0]], self.equ_w // 2, 1)
        pad_d = np.roll(e_img[[-1]], self.equ_w // 2, 1)
        e_img = np.concatenate([e_img, pad_d, pad_u], 0)
        # pad_l = e_img[:, [0]]
        # pad_r = e_img[:, [-1]]
        # e_img = np.concatenate([e_img, pad_l, pad_r], 1)

        return map_coordinates(e_img, [self.coor_y, self.coor_x],
                               order=order, mode='wrap')[..., 0]

    def run(self, equ_img:np.ndarray, equ_dep=None):

        h, w = equ_img.shape[:2]
        if h != self.equ_h or w != self.equ_w:
            equ_img = cv2.resize(equ_img, (self.equ_w, self.equ_h))
            if equ_dep is not None:
                equ_dep = cv2.resize(equ_dep, (self.equ_w, self.equ_h), interpolation=cv2.INTER_NEAREST)

        cube_img = np.stack([self.sample_equirec(equ_img[..., i], order=1)
                             for i in range(equ_img.shape[2])], axis=-1)

        if equ_dep is not None:
            cube_dep = np.stack([self.sample_equirec(equ_dep[..., i], order=0)
                                 for i in range(equ_dep.shape[2])], axis=-1)
            cube_dep = cube_dep * self.cosmaps

        if equ_dep is not None:
            return cube_img, cube_dep
        else:
            return cube_img

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw_dataset_path = args.input_folder
    output_path = args.output_folder

    cubemap_img_resolution = args.cubemap_resolution
    method_str = args.test_method
    if method_str == 'ctrlroom':
        split_lst = ['bedroom', 'livingroom']
    num_subviews = 8
    rotation_interval = 45
    sub_view_fov = 90
    subview_img_resolution = 512

    
    # e2c_fn = Equirec2Cube()
    for split in split_lst:
        split_folderpath = os.path.join(raw_dataset_path, split)
        scene_folders_lst = [f for f in os.listdir(split_folderpath) if os.path.isdir(os.path.join(split_folderpath, f)) and f.isdigit()]
        output_split_folder = os.path.join(output_path, split)
        os.makedirs(output_split_folder, exist_ok=True)

        for scene_folder in scene_folders_lst:
            scene_folderpath = os.path.join(split_folderpath, scene_folder)
            pano_img_path = glob.glob(os.path.join(scene_folderpath, '*_pano.png'))[0]
            # 6 cubemap view images
            save_cube_img_folderpathj = os.path.join(scene_folderpath, 'cubemap_img')
            os.makedirs(save_cube_img_folderpathj, exist_ok=True)

            # 8 subview images for MVDiffusion
            save_subview_img_folderpath = os.path.join(scene_folderpath, 'mvd_img')
            os.makedirs(save_subview_img_folderpath, exist_ok=True)

            pano_img = np.array(Image.open(pano_img_path).convert('RGB'))
            # print(f'cubemap_imgs.shape: {cubemap_imgs.shape}')
            img_name = os.path.basename(pano_img_path)
            print(f'process image: {pano_img_path}')
    
            e2c_fn = Equirectangular(pano_img_path)
            theta_angle_lst = [-90, 270, 0, 90, 180, -90]
            phi_angle_lst = [90, 0, 0, 0, 0, -90]

            cubemap_img_lst = []
            for i in range(6):
                subview_img = e2c_fn.GetPerspective(FOV=sub_view_fov, THETA=theta_angle_lst[i], 
                                                    PHI=phi_angle_lst[i], 
                                                    height=cubemap_img_resolution, 
                                                    width=cubemap_img_resolution)
                # print(f'cubemap_img.shape: {cubemap_img.shape}')
                cubemap_img_lst.append(subview_img)
                save_path = os.path.join(save_cube_img_folderpathj, f'{img_name[:-9]}_skybox{i}_sami.png')
                # print(f'save_path: {save_path}')
                Image.fromarray(subview_img).save(save_path)

            # warp 8 subview images
            init_degree = 0
            for i in range(num_subviews):
                _degree = (init_degree+rotation_interval*i) % 360
                img = warp_img(
                    fov=sub_view_fov, theta=_degree, phi=0, images=cubemap_img_lst, vx=theta_angle_lst, vy=phi_angle_lst)
                img = cv2.resize(img, (subview_img_resolution, subview_img_resolution))
                save_path = os.path.join(save_subview_img_folderpath, f'{img_name[:-4]}_{int(_degree)}.png')
                # print(f'save_path: {save_path}')
                Image.fromarray(img).save(save_path)
                
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder',
                        type=str,
                        default='/mnt/nas_3dv/hdd1/datasets/fangchuan/codes/HolisticDiffuScene/sample_results/pano_gen_experiments/')
    parser.add_argument('--output_folder',
                        type=str,
                        default='/mnt/nas_3dv/hdd1/datasets/fangchuan/codes/HolisticDiffuScene/sample_results/pano_gen_experiments/')
    parser.add_argument('--cubemap_resolution',
                        type=int,
                        default=512)
    parser.add_argument('--test_method',
                        type=str,
                        default='ctrlroom')
    args = parser.parse_args()
    main(args)