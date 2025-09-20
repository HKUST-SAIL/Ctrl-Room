"""
Train a diffusion model on images.
"""
import os
import sys

sys.path.append(".")  # Adds higher directory to python modules path.
sys.path.append("..")  # Adds higher directory to python modules path.
import argparse
import datetime

import torch
from mpi4py import MPI
from torch.utils.data import DataLoader

from src.diffu_layout.improved_diffusion import dist_util, logger
from src.diffu_layout.improved_diffusion.resample import create_named_schedule_sampler
from src.diffu_layout.improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from src.diffu_layout.improved_diffusion.train_util import TrainLoop
from dataset.st3d_dataset import ST3DDataset


def make_dataloader_cycle(iterable):
    while True:
        yield from iterable


def main():
    args = create_argparser().parse_args()

    # set up distributed training and logging
    dist_util.setup_dist()
    # log_dir = os.path.join(args.log_dir, datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f"))
    log_dir = args.log_dir
    logger.configure(dir=log_dir, format_strs=['tensorboard', 'stdout'])
    logger.set_level(logger.INFO)

    logger.log("creating UNet model and diffusion model...")
    unet_model, diffusion_model = create_model_and_diffusion(**args_to_dict(args,
                                                                            model_and_diffusion_defaults().keys()))
    unet_model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion_model)

    logger.log("creating data loader...")
    train_dataset = ST3DDataset(root_dir=args.data_dir,
                                max_text_sentences=4,
                                device=dist_util.dev(),
                                shard=MPI.COMM_WORLD.Get_rank(),
                                num_shards=MPI.COMM_WORLD.Get_size(),
                                random_text_desc=False,
                                use_gpt_text_desc=args.use_gpt_text_desc,
                                train_stats_file=args.dataset_stats_file,)
    logger.info(f"train_dataset length: {len(train_dataset)}")
    train_loader = DataLoader(train_dataset, 
                              batch_size=args.batch_size, 
                              shuffle=True, 
                              num_workers=0, 
                              drop_last=True,
                              pin_memory=True)

    logger.log("training...")
    TrainLoop(
        model=unet_model,
        diffusion=diffusion_model,
        # data=efficient_train_loader,
        data=make_dataloader_cycle(train_loader),
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    """ create argparser for data, model, and training configuration

    Returns:
        parser: _description_
    """
    defaults = dict(
        data_dir="/mnt/nas_3dv/hdd1/datasets/Structured3d/preprocessed/all_raw_light/bedroom",
        log_dir='log',
        schedule_sampler="loss-second-moment",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=100000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        dataset_stats_file=None,
        use_gpt_text_desc=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    # automatically add arguments
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
