import torch
import torch.nn as nn

import torchvision.transforms as T
import torchvision.transforms.functional as TF
from transformers import CLIPTokenizer, CLIPTextModel

# import clip

from . import logger

# To control logging level for various modules used in the application:
import logging
import re


def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)


class AbstractEncoder(nn.Module):

    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


# class CLIP(nn.Module):

#     def __init__(self, device, **kwargs):
#         super().__init__()

#         self.device = device
#         self.clip_model, self.clip_preprocess = clip.load("ViT-B/16", device=self.device, jit=False)
#         for param in self.clip_model.parameters():
#             param.requires_grad = False
#         self.aug = T.Compose([
#             T.Resize((224, 224)),
#             T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
#         ])

#     def get_text_embeds(self, prompt, **kwargs):

#         text = clip.tokenize(prompt).to(self.device)
#         print(f'text token.shape: {text.shape}')
#         text_z = self.clip_model.encode_text(text).last_hidden_state
#         print(f'text_z.shape: {text_z.shape}')
#         text_z = text_z / text_z.norm(dim=-1, keepdim=True)

#         return text_z

#     def get_img_embeds(self, image, **kwargs):

#         image_z = self.clip_model.encode_image(self.aug(image))
#         image_z = image_z / image_z.norm(dim=-1, keepdim=True)

#         return image_z

#     def train_step(self, clip_z, pred_rgb, grad_scale=10, **kwargs):
#         """
#             Args:
#                 grad_scale: scalar or 1-tensor of size [B], i.e. 1 grad_scale per batch item.
#         """
#         # TODO: resize the image from NeRF-rendered resolution (e.g. 128x128) to what CLIP expects (512x512), to prevent Pytorch warning about `antialias=None`.
#         image_z = self.clip_model.encode_image(self.aug(pred_rgb))
#         image_z = image_z / image_z.norm(dim=-1, keepdim=True)  # normalize features

#         loss = 0
#         if 'image' in clip_z:
#             loss -= ((image_z * clip_z['image']).sum(-1) * grad_scale).mean()

#         if 'text' in clip_z:
#             loss -= ((image_z * clip_z['text']).sum(-1) * grad_scale).mean()

#         return loss


class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from huggingface)"""
    LAYERS = ["last", "pooled", "hidden"]

    def __init__(self,
                 version="openai/clip-vit-large-patch14",
                 device="cuda",
                 max_length=77,
                 freeze=True,
                 layer="last",
                 layer_idx=None):  # clip-vit-base-patch32
        super().__init__()
        assert layer in self.LAYERS
        set_global_logging_level(logging.ERROR, ["transformers"])

        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version).to(device)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = layer_idx
        if layer == "hidden":
            assert layer_idx is not None
            assert 0 <= abs(layer_idx) <= 12

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def get_text_embeds(self, text):
        batch_encoding = self.tokenizer(text,
                                        truncation=True,
                                        max_length=self.max_length,
                                        return_length=True,
                                        return_overflowing_tokens=False,
                                        padding="max_length",
                                        return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens, output_hidden_states=self.layer == "hidden")
        if self.layer == "last":
            z = outputs.last_hidden_state
        elif self.layer == "pooled":
            z = outputs.pooler_output[:, None, :]
        else:
            z = outputs.hidden_states[self.layer_idx]
        return z

    def encode(self, text):
        return self(text)