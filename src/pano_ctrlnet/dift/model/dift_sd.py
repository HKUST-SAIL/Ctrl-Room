import os
import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Callable, Dict, List, Tuple, Optional, Union
from PIL import Image

import config

from cldm.cldm import ControlLDM
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.modules.diffusionmodules.util import timestep_embedding

import einops

class DIFTControlledUnetModel(UNetModel):
    def forward(self, x:torch.Tensor, timesteps:torch.Tensor=None, context:torch.Tensor=None, control:torch.Tensor=None, only_mid_control:bool=False, **kwargs):
        """ ControlNet forward pass, with optional control input, return estimation of noise and intermediate features

        Args:
            x (torch.Tensor): input noised latent code
            timesteps (torch.Tensor, optional): input timesteps. Defaults to None.
            context (torch.Tensor, optional): input text embedding. Defaults to None.
            control (torch.Tensor, optional): input features of control(hint) image. Defaults to None.
            only_mid_control (bool, optional): if only add control image to middle block of UNet. Defaults to False.

        Returns:
            _type_: _description_
        """
        hs = []
        # with torch.no_grad():
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)

        if control is not None:
            h += control.pop()

        # up block features
        up_blocks_features_dict = {}
        up_blocks_layer_idx_lst = [1,2,3,4,5,6,7,8,9,10]

        # print(f'DIFTControlledUnetModel::forward: len(self.output_blocks): {len(self.output_blocks)}')
        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                control_feats = control.pop()
                h = torch.cat([h, hs.pop() + control_feats], dim=1)
                # print(f'DIFTControlledUnetModel::forward: up_block_{i} control_features: {control_feats.shape}')
            h = module(h, emb, context)
            # print(f'DIFTControlledUnetModel::forward: up_block_{i}: {h.shape}')
            if i in up_blocks_layer_idx_lst:
                up_blocks_features_dict[i] = h

        h = h.type(x.dtype)

        # print(f'up_blocks_features_dict: {up_blocks_features_dict[up_blocks_layer_idx_lst[0]].shape}')
        return self.out(h), up_blocks_features_dict


class OneStepControlLDM(ControlLDM):

    def __init__(self, control_stage_config, control_key, only_mid_control, global_average_pooling=False, *args, **kwargs):
        super().__init__(control_stage_config, control_key, only_mid_control, global_average_pooling, *args, **kwargs)
        # for param in self.control_model.parameters():
        #     param.requires_grad = False
        # for param in self.self.controlnet_model.diffusion_model.parameters():
        #     param.requires_grad = False

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        control = batch[self.control_key]
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()
        return x, dict(c_crossattn=[c], c_concat=[control])

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        # UNet model
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['c_crossattn'], 1)

        if cond['c_concat'] is None:
            eps, _ = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:
            control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt)
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            if self.global_average_pooling:
                control = [torch.mean(c, dim=(2, 3), keepdim=True) for c in control]
            eps, up_block_dift_features_dict = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)

        # print(f'up_block_dift_features_dict: {up_block_dift_features_dict}')
        return eps, up_block_dift_features_dict

from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import random 
from pytorch_lightning import seed_everything
from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, extract_into_tensor
from annotator.util import resize_image, HWC3

class OneStepControlNetPipeline:
    def __init__(self, ckpt_filepath:str='/mnt/nas_3dv/hdd1/datasets/fangchuan/codes/Layout_Controlnet/ckpts/control_v11p_sd15_seg_livingroom_fullres_40000.ckpt', 
                 config_filepath:str='/mnt/nas_3dv/hdd1/fangchuan/Layout_Controlnet/models/dift_control_v11p_sd15_seg.yaml',
                 num_ddim_timesteps:int=50):
        model_name = 'control_v11p_sd15_seg'
        self.controlnet_model = create_model(config_filepath).cpu()
        self.controlnet_model.load_state_dict(load_state_dict(ckpt_filepath, location='cuda'), strict=False)
        self.controlnet_model = self.controlnet_model.cuda()
        self.ddim_sampler = DDIMSampler(self.controlnet_model)
        self.ddim_timesteps_lst = make_ddim_timesteps(ddim_discr_method='uniform', num_ddim_timesteps=num_ddim_timesteps,
                                                  num_ddpm_timesteps=1000,verbose=False)
        print(f'DDIM timesteps: {self.ddim_timesteps_lst}')
        self.ddim_sampler.make_schedule(ddim_num_steps=num_ddim_timesteps, ddim_eta=0.0, verbose=False)

    def process_inpainting(self,
                input_rgb_image:np.ndarray,
                input_sem_image:np.ndarray,
                prompt:str,
                a_prompt:str,
                n_prompt:str,
                num_samples:int,
                image_resolution:int,
                ddim_steps:int,
                guess_mode:bool,
                strength:float,
                scale:float,
                seed:int,
                eta:float,
                output_folder:str,
                edit_mask:torch.Tensor=None):

        os.makedirs(output_folder, exist_ok=True)

        with torch.no_grad():
            input_sem_image = HWC3(input_sem_image)
            input_rgb_image = HWC3(input_rgb_image)

            raw_img = resize_image(input_rgb_image.copy(), image_resolution)
            raw_img = torch.from_numpy(raw_img).float().cuda() / 127.0 - 1.0
            x_0 = torch.stack([raw_img for _ in range(num_samples)], dim=0)
            x_0 = einops.rearrange(x_0, 'b h w c -> b c h w').clone()
            x_0 = self.controlnet_model.get_first_stage_encoding(self.controlnet_model.encode_first_stage(x_0))

            control_img = resize_image(input_sem_image.copy(), image_resolution)
            H, W, C = control_img.shape
            control = torch.from_numpy(control_img.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()
            print(f'control_img.shape: {control.shape}')

            edit_mask = edit_mask.to(self.controlnet_model.device)

            if seed == -1:
                seed = random.randint(0, 65535)
            seed_everything(seed)

            if config.save_memory:
                self.controlnet_model.low_vram_shift(is_diffusing=False)

            cond = {
                "c_concat": [control],
                "c_crossattn": [self.controlnet_model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]
            }
            un_cond = {
                "c_concat": None if guess_mode else [control],
                "c_crossattn": [self.controlnet_model.get_learned_conditioning([n_prompt] * num_samples)]
            }
            shape = (num_samples, 4, H // 8, W // 8)

            if config.save_memory:
                self.controlnet_model.low_vram_shift(is_diffusing=True)

            self.controlnet_model.control_scales = [strength * (0.825**float(12 - i)) for i in range(13)] if guess_mode else ([strength] *
                                                                                                            13)
            # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            
            # sampled latent shape: (1, 4, 64, 128)
            samples, intermediates = self.ddim_sampler.ddim_sampling_hacked(
                                                        cond=cond,
                                                        shape=shape,
                                                        x0=x_0,
                                                        mask=edit_mask,
                                                        x_T=None,
                                                        unconditional_guidance_scale=scale,
                                                        unconditional_conditioning=un_cond,
                                                        mix_mask_unmask=True,)

            if config.save_memory:
                self.controlnet_model.low_vram_shift(is_diffusing=False)

            x_samples = self.controlnet_model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(
                0, 255).astype(np.uint8)

            # save samples
            results = [x_samples[i] for i in range(num_samples)]

        # return torch.from_numpy(control_img).to(self.controlnet_model.device), results[0], samples[0]
        return results[0], samples[0]

    def process(self,input_control_image, input_rgb_image, prompt, a_prompt, n_prompt, num_samples, guess_mode, strength, seed, 
                dift_timestep:int=261, 
                dift_up_blocks_layer_idx:int=5):

        with torch.no_grad():

            input_control_img = input_control_image.copy()
            H, W, C = input_control_img.shape

            control = torch.from_numpy(input_control_img.copy()).float().to(self.controlnet_model.device) / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()

            input_rgb = torch.from_numpy(input_rgb_image.copy()).float().to(self.controlnet_model.device)
            input_rgb = (input_rgb / 127.5) - 1.0
            input_rgb = torch.stack([input_rgb for _ in range(num_samples)], dim=0)
            input_rgb = einops.rearrange(input_rgb, 'b h w c -> b c h w')
            encoder_posterior = self.controlnet_model.encode_first_stage(input_rgb)
            z0 = self.controlnet_model.get_first_stage_encoding(encoder_posterior)


            if seed == -1:
                seed = random.randint(0, 65535)
            seed_everything(seed)

            cond = {
                "c_concat": [control],
                "c_crossattn": [self.controlnet_model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]
            }
            un_cond = {
                "c_concat": None if guess_mode else [control],
                "c_crossattn": [self.controlnet_model.get_learned_conditioning([n_prompt] * num_samples)]
            }
            shape = (4, H // 8, W // 8)

            self.controlnet_model.control_scales = [strength * (0.825**float(12 - i)) for i in range(13)] if guess_mode else ([strength] *
                                                                                                            13)
            # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            assert dift_timestep in list(self.ddim_timesteps_lst), "dift_timestep should be in ddim_timesteps_lst"
            ts_index = list(self.ddim_timesteps_lst).index(dift_timestep)
            ts = torch.full((num_samples,), dift_timestep, dtype=torch.long, device=self.controlnet_model.device)
            z_t = self.controlnet_model.q_sample(z0, ts)
            # print(f'latents_noisy shape: {z_t.shape}')

            # only one step denoising
            # est_eps, dift_up_blocks_feats_dict = self.controlnet_model.apply_model(z_t, ts, cond)
            z_t_minus_1, pred_z0, dift_up_blocks_feats_dict = self.denoise_step( x_t=z_t, t=ts,t_index=ts_index, cond=cond, un_cond=un_cond)

            unet_dift_feats = dift_up_blocks_feats_dict[dift_up_blocks_layer_idx]
            unet_ft = unet_dift_feats.mean(0, keepdim=True) # 1,c,h,w
            # print(f'unet_ft shape: {unet_ft.shape}')

        return z0.detach(), z_t.detach(), unet_ft.detach()

    def denoise_step(self, x_t, t, t_index, cond, un_cond, un_cond_scale=9.0):
        """ Denoise a single timestep

        Args:
            x_t (torch.Tensor): input noised latent code
            t (int): timestep
            cond (dict): conditioning and control information

        Returns:
            _type_: x_(t-1), x_0, unet_features_dict
        """
        x_t_minus_1, pred_x0, dift_up_blocks_feats_dict = self.ddim_sampler.p_sample_ddim_with_unet_feat(x=x_t, c=cond, t=t, index=t_index, use_original_steps=False,
                                    quantize_denoised=False, temperature=1.,
                                    noise_dropout=0., score_corrector=None,
                                    corrector_kwargs=None,
                                    unconditional_guidance_scale=un_cond_scale,
                                    unconditional_conditioning=un_cond,
                                    dynamic_threshold=None)
        return (x_t_minus_1, pred_x0, dift_up_blocks_feats_dict)

if __name__ == '__main__':
    input_pano_a_filepath = '/mnt/nas_3dv/hdd1/datasets/fangchuan/codes/Layout_Controlnet/sample_results/openai-2023-08-16-19-31-43-309675/0/0.png'
    input_seg_a_filepath = '/mnt/nas_3dv/hdd1/datasets/fangchuan/codes/Layout_Controlnet/sample_results/openai-2023-08-16-19-31-43-309675/0/raw_sem_bbox_img.png'
    input_pano_b_filepath = '/mnt/nas_3dv/hdd1/datasets/fangchuan/codes/Layout_Controlnet/sample_results/openai-2023-08-16-19-31-43-309675/0/1.png'
    input_seg_b_filepath = '/mnt/nas_3dv/hdd1/datasets/fangchuan/codes/Layout_Controlnet/sample_results/openai-2023-08-16-19-31-43-309675/0/edited_sem_bbox_img.png'
    
    rgb_img_lst = []
    seg_img_lst = []

    pano_a_img = Image.open(input_pano_a_filepath).convert('RGB')
    pano_a_img = np.array(pano_a_img)
    rgb_img_lst.append(pano_a_img)

    seg_a_img = Image.open(input_seg_a_filepath)
    seg_a_img = np.array(seg_a_img)
    seg_img_lst.append(seg_a_img)

    pano_b_img = Image.open(input_pano_b_filepath).convert('RGB')
    pano_b_img = np.array(pano_b_img)
    rgb_img_lst.append(pano_b_img)

    seg_b_img = Image.open(input_seg_b_filepath)
    seg_b_img = np.array(seg_b_img)
    seg_img_lst.append(seg_b_img)

    num_samples = 1
    text_prompt = 'The living room has four walls. The room has a sofa and a cabinet . There is a lamp above the sofa . There is a second lamp to the right of the first lamp .'
    controlnet_pipeline = OneStepControlNetPipeline()

    for rgb_img, seg_img in zip(rgb_img_lst, seg_img_lst):
        
        input_params_dict = dict(
                            input_rgb_image = rgb_img,
                            input_control_image = seg_img,
                            prompt = text_prompt,
                            a_prompt = 'best quality',
                            n_prompt = 'lowres, bad anatomy, bad hands, cropped, worst quality',
                            num_samples = num_samples,
                            guess_mode = False,
                            strength = 1.0,
                            seed = 12345,
                            dift_timestep=261)
        controlnet_pipeline.process(**input_params_dict)
