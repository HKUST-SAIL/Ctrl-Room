import argparse
import os
import json
import torch
import numpy as np
from PIL import Image
from torchmetrics.image.inception import InceptionScore


def pil_to_torch(img, device, normalize=True):
    img = torch.tensor(np.array(img), device=device).permute(2, 0, 1)
    if normalize:
        img = img.float()
        img = img / 255.0
    return img


def main(image_folder):
    inception_score = InceptionScore(normalize=True).cuda()
    images = []
    scene_list = [f for f in os.listdir(image_folder) if os.path.isdir(os.path.join(image_folder, f))]
    for i in scene_list:
        renderings_folder = os.path.join(image_folder, i, 'model_raw/render_output/image')
        renderings_folder = os.path.join(image_folder, i, 'render_output/image')
        image_list = sorted(os.listdir(renderings_folder))
        for j in image_list:
            images.append(Image.open(os.path.join(renderings_folder, j)).convert('RGB'))
    images = torch.stack([pil_to_torch(i, inception_score.device, normalize=True) for i in images], dim=0)
    print(len(images))

    inception_score.update(images)
    out = inception_score.compute()

    out_dict = {
        "mean": out[0].cpu().numpy().item(),
        "std": out[1].cpu().numpy().item(),
    }
    
    print(out_dict)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # GENERAL CONFIG
    image_folder = '/mnt/nas_3dv/hdd1/fangchuan/mesh_generation_experiments/text2light_results'

    main(image_folder)