import os
import sys
sys.path.append('.')
sys.path.append('..')


import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from src.pano_ctrlnet.dataset.st3d_dataset import ST3DDataset
from src.pano_ctrlnet.cldm.logger import ImageLogger
from src.pano_ctrlnet.cldm.model import create_model, load_state_dict

def register_user_signals(trainer: pl.Trainer):
    ckptdir = trainer.checkpoint_callback.dirpath

    # allow checkpointing via USR1
    def melk(*args, **kwargs):
        # run all checkpoint hooks
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def divein(*args, **kwargs):
        if trainer.global_rank == 0:
            import pudb
            pudb.set_trace()

    import signal

    signal.signal(signal.SIGUSR1, melk)
    signal.signal(signal.SIGUSR2, divein)

# Configs

batch_size = 10
logger_freq = 1000
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model_name = './src/pano_ctrlnet/models/control_v11p_sd15_seg.yaml'
model = create_model(f'{model_name}').cpu()
model.load_state_dict(load_state_dict('../ckpts/control_v11p_sd15_seg_40000.ckpt', location='cuda'), strict=False)
# model.load_state_dict(load_state_dict(f'../ckpts/{model_name}.pth', location='cuda'), strict=False)
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# dataset params
b_flip = True
b_rotate = True
b_gamma = False
use_ddp = True
room_type_str = 'kitchen'
dataset_path = '/path/to//'

train_dataset = ST3DDataset(root_dir=dataset_path,
                           split='train',
                           room_type=room_type_str,
                           flip=b_flip,
                           rotate=b_rotate,
                           gamma=b_gamma,
                           stretch=True,
                           downsample_scale=1,
                           shard=0,
                           num_shards=1)
test_dataset = ST3DDataset(root_dir=dataset_path,
                           split='test',
                           room_type=room_type_str,
                           flip=False,
                           rotate=False,
                           gamma=False,
                           stretch=False,
                           downsample_scale=1,
                           shard=0,
                           num_shards=1)
print(f"train_dataset length: {len(train_dataset)}")
print(f"test_dataset length: {len(test_dataset)}")

train_dataloader = DataLoader(train_dataset, num_workers=2, batch_size=batch_size, shuffle=True)
# test_dataloader = DataLoader(test_dataset, num_workers=2, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)

# training params
min_epochs = None
max_epochs = 1000
min_steps = None
max_steps = 1

trainer = pl.Trainer(accelerator='gpu',
                     gpus=2,
                     precision=32,
                     callbacks=[logger],
                     strategy='ddp',
                     min_epochs=min_epochs,
                     max_epochs=max_epochs)

# register user signals
register_user_signals(trainer)

# zero train
trainer.fit(model, train_dataloader)
