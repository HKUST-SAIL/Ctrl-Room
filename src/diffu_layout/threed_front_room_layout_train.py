"""
Train a diffusion model on images.
"""
import os
import sys

sys.path.append(".")  # Adds higher directory to python modules path.
sys.path.append("..")  # Adds higher directory to python modules path.
import argparse
import datetime

from mpi4py import MPI
from torch.utils.data import DataLoader

from improved_diffusion import dist_util, logger
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop
from dataset.threed_front_dataset_diffuscene import ThreedFrontDataset
from utils.utils import load_config


def make_dataloader_cycle(iterable):
    while True:
        yield from iterable


def load_data(args, room_type, config, is_train=True):

    # load training data
    train_dataset = ThreedFrontDataset(
        room_type=room_type,
        is_train=is_train,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        config=config,
    )
    logger.info(f"train_dataset length: {len(train_dataset)}")
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=0,
                              pin_memory=True,
                              drop_last=True)

    while True:
        yield from train_loader


def main():
    args = create_argparser().parse_args()

    # Parse the config file
    config = load_config(args.config_file)
    args.dataset_stats_file = config['data']['train_stats']

    # set up distributed training and logging
    dist_util.setup_dist()
    # log_dir = os.path.join(args.log_dir, datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f"))
    log_dir = args.log_dir
    logger.configure(dir=log_dir, format_strs=['tensorboard', 'stdout', 'log'])
    logger.set_level(logger.INFO)

    logger.log("creating UNet model and diffusion model...")
    unet_model, diffusion_model = create_model_and_diffusion(**args_to_dict(args,
                                                                            model_and_diffusion_defaults().keys()))
    unet_model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion_model)

    logger.log("creating data loader...")
    if args.room_type == 'bedroom':
        room_type = 'bedroom'
    elif args.room_type == 'livingroom':
        room_type = 'living room'
    elif args.room_type == 'diningroom':
        room_type = 'dining room'

    # load training data
    data = load_data(args, room_type, config, is_train=True)

    logger.log("training...")
    TrainLoop(
        model=unet_model,
        diffusion=diffusion_model,
        data=data,
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
        log_dir='log',
        schedule_sampler="loss-second-moment",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        room_type="bedroom",
        config_file="../config/3dfront_bedroom_config.yaml",
        dataset_stats_file=None,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    # automatically add arguments
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
