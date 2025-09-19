import torch
from PIL import Image
import numpy as np
# for clip_score
from torchmetrics.functional.multimodal import clip_score
from functools import partial

# 加载CLIP模型
clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")


def calculate_clip_score(images, prompts):
    clip_score = clip_score_fn(torch.from_numpy(images).permute(0, 3, 1, 2), prompts).detach()
    return round(float(clip_score), 4)


SCENE_TEXT_MAP = {
    'bedroom_scene_03011_13456': 'The bedroom has eight walls. The room has two windows and a bed .',
    'bedroom_scene_03084_641793': 'The bedroom has four walls. The room has a window and a sofa . The table is to the left of the sofa',
    'bedroom_scene_03113_560': 'The bedroom has six walls. The room has a cabinet and a window',
    'bedroom_scene_03243_800738': 'The bedroom has four walls. The room has a window and a picture',
    'bedroom_scene_03422_794742': 'The bedroom has four walls. The room has a cabinet and a window ',
    'livingroom_scene_03027_2723': 'The living room has fourteen walls. The room has a cabinet and a chair ', 
    'livingroom_scene_03049_284331': 'The living room has eight walls. The room has a picture , a shelves and a cabinet ', 
    'livingroom_scene_03125_536': 'The living room has twelve walls. The room has a cabinet and a window . There is a lamp above the cabinet ', 
    'livingroom_scene_03200_522357': 'The living room has six walls. The room has a cabinet , a fridge and a window . There is a chair to the left of the fridge . There is a second chair behind the first chair', 
    'livingroom_scene_03300_190736': 'The living room has ten walls. The room has a picture and a window ', 
    'livingroom_scene_03309_12023': 'The living room has ten walls. The room has a cabinet , a fridge and a shelves . The fridge is to the right of the cabinet . There is a chair to the left of the shelves ', 
}

import os
from glob import glob
import os.path as osp
import numpy as np

b_test_our_method = False
INPUT_FOLDER_DICTS = ['/mnt/nas_3dv/hdd1/fangchuan/mesh_generation_experiments/our_results/',
                      '/mnt/nas_3dv/hdd1/fangchuan/mesh_generation_experiments/text2light_results/',
                      '/mnt/nas_3dv/hdd1/fangchuan/mesh_generation_experiments/blockade_results/',
                      '/mnt/nas_3dv/hdd1/fangchuan/mesh_generation_experiments/text2room_results/',
                      '/mnt/nas_3dv/hdd1/fangchuan/mesh_generation_experiments/mvdiffusion_results/',]
if __name__ == '__main__':
    input_folderpath = INPUT_FOLDER_DICTS[0] if b_test_our_method else INPUT_FOLDER_DICTS[1]
    input_subfolder_lst = [f for f in os.listdir(input_folderpath) if osp.isdir(osp.join(input_folderpath, f))]

    sum_score = 0
    invalid_scene_cnt = 0
    for i, sub_folder in enumerate(input_subfolder_lst):

        if sub_folder not in SCENE_TEXT_MAP.keys():
            print('Invalid sub_folder: ', sub_folder)
            invalid_scene_cnt +=1
            continue

        text_prompt = SCENE_TEXT_MAP[sub_folder]

        scene_folderpath = osp.join(input_folderpath, sub_folder)
        if b_test_our_method:
            rendering_folderpath = osp.join(scene_folderpath, 'model_raw', 'render_output', 'image')
        else:
            rendering_folderpath = osp.join(scene_folderpath, 'render_output', 'image')

        imgs_lst = sorted(glob(osp.join(rendering_folderpath, '*.png')), key=lambda x: int(x.split('/')[-1].split('.')[0]))
        # print(f'image_files_lst: {imgs_lst}')

        batch_imgs = []
        for img_path in imgs_lst:
            img = Image.open(img_path).convert('RGB')
            img = np.array(img)
            batch_imgs.append(img)
        
        text_prompt_lst = [text_prompt] * len(batch_imgs)

        batch_imgs = np.stack(batch_imgs, axis=0)
        # print(f'batch_imgs.shape: {batch_imgs.shape}')

        # calculate clip score
        score = calculate_clip_score(batch_imgs, text_prompt_lst)
        print(f'scene {sub_folder} clip_score: {score}')
        sum_score += score
    
    print(f'average clip_score: {sum_score / (len(input_subfolder_lst) - invalid_scene_cnt)}')