import os
import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import cv2
import numpy as np

import torch
from utils import warp_img, Equirectangular
from PIL import Image
from tqdm import tqdm
from lavis.models import load_model_and_preprocess

from scipy.ndimage import map_coordinates



def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw_dataset_path = args.input_folder
    output_path = args.output_folder

    cubemap_img_resolution = args.cubemap_resolution
    split_lst = ['train/bedroom', 'test/bedroom', 'train/livingroom', 'test/livingroom']
    num_subviews = 8
    rotation_interval = 45
    sub_view_fov = 90
    subview_img_resolution = 512

    # loads BLIP caption base model, with finetuned checkpoints on MSCOCO captioning dataset.
    # this also loads the associated image processors
    caption_model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)
    
    for split in split_lst:
        split_folderpath = os.path.join(raw_dataset_path, split)
        pano_img_folderpath = os.path.join(split_folderpath, 'img')

        output_split_folder = os.path.join(output_path, split)
        os.makedirs(output_split_folder, exist_ok=True)

        # 6 cubemap view images
        save_cube_img_folderpathj = os.path.join(output_split_folder, 'cubemap_img')
        os.makedirs(save_cube_img_folderpathj, exist_ok=True)

        # 8 subview images for MVDiffusion
        save_subview_img_folderpath = os.path.join(output_split_folder, 'mvd_img')
        os.makedirs(save_subview_img_folderpath, exist_ok=True)

        save_subview_text_folderpath = os.path.join(output_split_folder, 'mvd_text')
        os.makedirs(save_subview_text_folderpath, exist_ok=True)

        pano_img_lst = [name for name in os.listdir(pano_img_folderpath) if name.endswith('.png')]
        for img_name in pano_img_lst:
            img_filepath = os.path.join(pano_img_folderpath, img_name)
            pano_img = np.array(Image.open(img_filepath).convert('RGB'))
            # cubemap_imgs = e2c_fn.run(pano_img)
            # print(f'cubemap_imgs.shape: {cubemap_imgs.shape}')
            print(f'process image: {img_filepath}')
    
            e2c_fn = Equirectangular(img_filepath)
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
                Image.fromarray(subview_img).save(os.path.join(save_cube_img_folderpathj, f'{img_name[:-4]}_skybox{i}_sami.png'))

            # warp 8 subview images
            init_degree = 0
            for i in range(num_subviews):
                _degree = (init_degree+rotation_interval*i) % 360
                img = warp_img(
                    fov=sub_view_fov, theta=_degree, phi=0, images=cubemap_img_lst, vx=theta_angle_lst, vy=phi_angle_lst)
                img = cv2.resize(img, (subview_img_resolution, subview_img_resolution))
                Image.fromarray(img).save(os.path.join(save_subview_img_folderpath, f'{img_name[:-4]}_{int(_degree)}.png'))

                img_pil = Image.fromarray(img)
                # preprocess the image
                # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
                image = vis_processors["eval"](img_pil).unsqueeze(0).to(device)
                # generate caption
                text_desc = caption_model.generate({"image": image})
                print(text_desc)
                with open(os.path.join(save_subview_text_folderpath, f'{img_name[:-4]}_{int(_degree)}.txt'), 'w') as f:
                    f.write(text_desc[0])
                
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder',
                        type=str,
                        default='/mnt/nas_3dv/hdd1/datasets/datasets/Structured3D/text2pano/')
    parser.add_argument('--output_folder',
                        type=str,
                        default='/mnt/nas_3dv/hdd1/datasets/datasets/Structured3D/mvdiffusion_text2pano/')
    parser.add_argument('--cubemap_resolution',
                        type=int,
                        default=512)
    args = parser.parse_args()
    main(args)