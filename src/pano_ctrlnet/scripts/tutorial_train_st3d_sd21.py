import os
import sys
sys.path.append('.')
sys.path.append('..')

from share import *

# from mpi4py import MPI

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset.st3d_dataset import ST3DDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


# Configs
resume_path = '../ckpts/control_st3d_sd21_ini.ckpt'
batch_size = 4
logger_freq = 1000
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('../models/cldm_v21.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# dataset params
b_flip = False
b_rotate = True
b_gamma = False
use_ddp = True
dataset_path = '/home/fangchuan/datasets/Structured3d/preprocessed/debug_text_emb/train/bedroom/'


train_dataset = ST3DDataset(root_dir=dataset_path, flip=b_flip, rotate=b_rotate, gamma=b_gamma, shard=0, num_shards=1)
print(f"train_dataset length: {len(train_dataset)}")

dataloader = DataLoader(train_dataset, num_workers=32, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)

# Train!
min_epochs = 1000
max_epochs = -1
min_steps = None
max_steps = -1
trainer = pl.Trainer(accelerator='gpu', gpus=2, precision=32, callbacks=[logger], strategy="ddp", min_epochs=min_epochs, max_epochs=max_epochs, min_steps=min_steps, max_steps=max_steps)

resume_ckpt_path = 'lightning_logs/version_3/checkpoints/epoch=6-step=7013.ckpt'
trainer.fit(model=model, train_dataloader = dataloader, ckpt_path=resume_ckpt_path)