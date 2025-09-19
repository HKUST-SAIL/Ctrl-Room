import os
import sys

sys.path.append('.')
sys.path.append('..')

import random
import datetime
import argparse
import glob
import gc
import json
from share import *
import config

import cv2
from PIL import Image, ImageOps
import einops
import gradio as gr
import numpy as np
import torch

from typing import List, Dict, Any

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

from annotator.oneformer.oneformer.data.datasets.register_ade20k_panoptic import ADE20K_150_CATEGORIES

ade_labels = [label_dict["name"] for label_dict in ADE20K_150_CATEGORIES]
# print(f'ade_labels: {ade_labels}')
ade_colors = [list(label_dict["color"]) for label_dict in ADE20K_150_CATEGORIES]


def load_pano_gen_model(ckpt_filepath:str, device: str = 'cuda'):
    model_name = 'control_v11p_sd15_seg'
    model = create_model(f'../models/{model_name}.yaml').cpu()
    model.load_state_dict(load_state_dict(ckpt_filepath, location=device), strict=False)
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)

    return model, ddim_sampler


samples_latent_lst = []
samples_img_lst = []
control_img_lst = []

def process(model,
            ddim_sampler:DDIMSampler,
            input_image,
            prompt,
            a_prompt,
            n_prompt,
            num_samples,
            image_resolution,
            ddim_steps,
            guess_mode,
            strength,
            scale,
            seed,
            eta,
            output_folder,
            init_latent=None):

    os.makedirs(output_folder, exist_ok=True)

    with torch.no_grad():
        input_image = HWC3(input_image)

        control_img = resize_image(input_image.copy(), image_resolution)
        H, W, C = control_img.shape

        control = torch.from_numpy(control_img.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()
        print(f'control_img.shape: {control.shape}')

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {
            "c_concat": [control],
            "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]
        }
        un_cond = {
            "c_concat": None if guess_mode else [control],
            "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]
        }
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825**float(12 - i)) for i in range(13)] if guess_mode else ([strength] *
                                                                                                          13)
        # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01

        x_0 = None
        x_T = init_latent
        edited_mask = None

        # sampled latent shape: (1, 4, 64, 128)
        samples, intermediates = ddim_sampler.sample(S=ddim_steps,
                                                     batch_size=num_samples,
                                                     shape=shape,
                                                     conditioning=cond,
                                                     x0=x_0,
                                                     mask=edited_mask,
                                                     x_T=x_T,
                                                     verbose=False,
                                                     eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond,
                                                     use_consistent_sampling=True,)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(
            0, 255).astype(np.uint8)

        # save samples
        print(f'x_samples.shape: {x_samples.shape}')
        # for i in range(num_samples):
        #     cv2.imwrite(os.path.join(current_sample_folder, f'{i}.png'), x_samples[i])

        results = [x_samples[i] for i in range(num_samples)]
        # samples_img_lst.append(results[0])
        # control_img_lst.append(control.cpu().numpy())

    return torch.from_numpy(control_img).to(model.device), results[0], samples[0],


def read_text_prompt(filepath: str):
    text_lst = []
    with open(filepath, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            text_lst.append(line.strip())
    return text_lst


def load_from_jsonl(filename: str):
    assert filename.endswith(".jsonl")
    if not os.path.exists(filename):
        return None

    data = []
    with open(filename, encoding="utf-8") as f:
        for row in f:
            data.append(json.loads(row))
    return data

def main(args):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load control images
    input_folders_lst = [
        f
        for f in os.listdir(args.input_folder)
        if os.path.isdir(os.path.join(args.input_folder, f))
    ]
    input_folders_lst = sorted(input_folders_lst, key=lambda x: int(x.split('_')[-1]))
    # load text prompt
    input_text_promt_filepath = os.path.join(args.input_folder, 'text_prompt.txt')
    if os.path.exists(input_text_promt_filepath):
        input_text_prompts_lst = read_text_prompt(input_text_promt_filepath)
    else:
        # SpatiailGen-Testset
        input_text_prompt_filepath = os.path.join(args.input_folder, 'test_split_caption.jsonl')
        input_text_prompts_lst = [list(scene_id_caption_dict.values())[0] for scene_id_caption_dict in load_from_jsonl(input_text_prompt_filepath)]
    # load panorama generation model
    pano_gen_model, ddim_sampler = load_pano_gen_model(ckpt_filepath=args.ckpt_filepath, device=device)

    for i, (text_prompt, img_folder) in enumerate(zip(input_text_prompts_lst, input_folders_lst)):
        img_folder = os.path.join(args.input_folder, img_folder)
        output_folder = img_folder

        img_path = glob.glob(os.path.join(img_folder, '*_sem.png'))[0]
        img_name = img_path.split('/')[-1]
        print(f'Predict panorama {i}/{len(input_folders_lst)}: {text_prompt}      {img_name}')

        sem_layout_pano_img = Image.open(img_path).convert('RGB')
        sem_layout_pano_img = np.array(sem_layout_pano_img)

        H, W, C = sem_layout_pano_img.shape

        batch_size = args.num_samples
        ddim_steps = args.ddim_steps
        control_strength = args.strength
        uncond_scale = args.scale
        seed = args.seed
        ddim_eta = args.ddim_eta

        input_params_dict = dict(
            model=pano_gen_model,
            ddim_sampler=ddim_sampler,
            input_image=sem_layout_pano_img,
            prompt=text_prompt,
            a_prompt='best quality',
            n_prompt='lowres, bad anatomy, bad hands, cropped, worst quality',
            num_samples=batch_size,
            image_resolution=H,
            ddim_steps=ddim_steps,
            guess_mode=False,
            strength=control_strength,
            scale=uncond_scale,
            seed=seed,
            eta=ddim_eta,
            output_folder=output_folder,
            init_latent=None,
        )

        # control_img: H,W,C
        # sampled_img: H,W,C
        # sampled_latent: H//8,W//8,4
        control_img, sampled_img, sampled_latent = process(**input_params_dict)

        control_img_lst.append(control_img)
        samples_img_lst.append(sampled_img)
        samples_latent_lst.append(sampled_latent)

        # save samples
        Image.fromarray(sampled_img).save(os.path.join(output_folder, f'{img_name[:-8]}_pano.png'))

    gc.collect()
    torch.cuda.empty_cache()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder',
                        type=str,
                        default='/home/fangchuan/codes/Structured3D/sample_results/openai-2023-09-08-21-45-44-129369/livingroom')
    parser.add_argument('--ckpt_filepath',
                        type=str,
                        default='/home/fangchuan/codes/Structured3D/ckpts/control_v11p_sd15_seg_livingroom_fullres_40000.ckpt')
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--image_resolution', type=int, default=512)
    parser.add_argument('--ddim_steps', type=int, default=200)
    parser.add_argument('--guess_mode', action='store_true')
    parser.add_argument('--strength', type=float, default=1.0)
    parser.add_argument('--scale', type=float, default=9.0)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--ddim_eta', type=float, default=0.0)
    args = parser.parse_args()
    main(args)