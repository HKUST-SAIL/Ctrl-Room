import argparse
import inspect

from typing import Dict, List, Tuple, Any
from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .unet import SuperResModel, UNetModel

from . import logger

NUM_CLASSES = 16
NUM_TEXT_EMBEDDING_DIM = 768


def model_and_diffusion_defaults():
    """
    Defaults for layout training.
    """
    return dict(
        # image_size=64,
        layout_size=32,
        layout_channels=23,
        num_channels=128,
        num_res_blocks=2,
        num_heads=8,
        num_heads_upsample=-1,
        # attention_resolutions="16,8",
        attention_resolutions="32, 16, 8, 4",
        dropout=0.0,
        b_learn_sigma=False,
        sigma_small=False,
        b_class_cond=True,
        b_text_cond=False,
        diffusion_steps=1000,
        noise_schedule="cosine",
        timestep_respacing="",
        b_use_kl=False,
        predict_xstart=False,
        rescale_timesteps=True,
        rescale_learned_sigmas=True,
        use_3d_iou=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        use_input_encoding=False,
        dataset_stats_file=None,
    )


def create_model_and_diffusion(
    layout_size: int,
    layout_channels: int,
    b_class_cond: bool,
    b_text_cond: bool,
    b_learn_sigma: bool,
    sigma_small,
    num_channels: int,
    num_res_blocks: int,
    num_heads: int,
    num_heads_upsample,
    attention_resolutions,
    dropout: float,
    diffusion_steps: int,
    noise_schedule,
    timestep_respacing,
    b_use_kl: bool,
    predict_xstart: bool,
    rescale_timesteps: bool,
    rescale_learned_sigmas: bool,
    use_3d_iou: bool,
    use_checkpoint: bool,
    use_scale_shift_norm: bool,
    use_input_encoding: bool,
    dataset_stats_file: str = None,
):
    """ create UNet model and diffusion, according to the given parameters

    Args:
        layout_size (int): Layout instance number
        layout_channels (int): Layout feature channel size
        b_class_cond (bool): whther use class labels as conditions
        b_text_cond (bool): whether use text as conditions
        b_learn_sigma (bool): whether to sigma
        sigma_small (_type_): _description_
        num_channels (int): UNet channel size
        num_res_blocks (int): number of residual blocks per downsample.
        num_heads (int): number of attention heads
        num_heads_upsample (_type_): _description_
        attention_resolutions (_type_):  which resnet will use attention
        dropout (float): dropout probability
        diffusion_steps (int): number of diffusion steps
        noise_schedule (str): use linear or cosine noise schedule
        timestep_respacing (_type_): _description_
        b_use_kl (bool): whether to use KL divergence
        predict_xstart (_type_): _description_
        rescale_timesteps (_type_): _description_
        rescale_learned_sigmas (bool): whether to rescale learned variance;
        use_3d_iou (bool): whether use 3d iou loss;
        use_checkpoint (_type_): _description_
        use_scale_shift_norm (_type_): _description_

    Returns:
        _type_: _description_
    """
    # create UNet
    model = create_model(
        layout_size=layout_size,
        layout_channels=layout_channels,
        num_channels=num_channels,
        num_res_blocks=num_res_blocks,
        learn_sigma=b_learn_sigma,
        class_cond=b_class_cond,
        text_cond=b_text_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        use_input_encoding=use_input_encoding,
    )
    # create diffusion process
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=b_learn_sigma,
        sigma_small=sigma_small,
        noise_schedule=noise_schedule,
        use_kl=b_use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        b_use_3d_iou=use_3d_iou,
        timestep_respacing=timestep_respacing,
        dataset_stats_file=dataset_stats_file,
    )
    return model, diffusion


def create_model(
    # image_size,
    layout_size,
    layout_channels,
    num_channels,
    num_res_blocks,
    learn_sigma,
    class_cond,
    text_cond,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
    use_input_encoding,
):
    layout_feature_size = layout_channels
    if layout_feature_size == 1024:
        channel_mult = (1, 2, 3, 4)
    elif layout_feature_size == 32:  # for st3d_bedroom
        channel_mult = (1, 2, 2, 2)
    elif layout_feature_size == 33 or layout_feature_size == 30:  # for 3d_front_bedroom
        channel_mult = (1, 2, 2, 2)
    elif layout_feature_size == 36 or layout_feature_size == 33:  # for 3d_front_livingroom
        channel_mult = (1, 2, 2, 2)
    elif layout_feature_size == 34:  # for st3d_livingroom
        channel_mult = (1, 2, 2, 2)
    elif layout_feature_size == 28:  # for st3d_kitchen
        channel_mult = (1, 2, 2, 2)
    else:
        raise ValueError(f"unsupported layout size: {layout_channels}")

    # assert class_cond != text_cond, "only one of class_cond and text_cond can be True"

    # downsample ratio at attention layers
    attention_downsample_ratio_lst = []
    for res in attention_resolutions.split(","):
        attention_downsample_ratio_lst.append(layout_feature_size // int(res))
    logger.info(f"attention downsaple ratios: {attention_downsample_ratio_lst}")

    return UNetModel(
        in_channels=layout_channels,
        model_channels=num_channels,
        out_channels=(layout_channels if not learn_sigma else 2 * layout_channels),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_downsample_ratio_lst),
        dropout=dropout,
        channel_mult=channel_mult,
        # num_classes=(NUM_CLASSES if class_cond else None),
        text_condition=text_cond,
        text_emb_dim=(NUM_TEXT_EMBEDDING_DIM if text_cond else None),
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        use_input_encoding=use_input_encoding,
        num_instances=layout_size,
        instance_emb_dim=128,
    )


def sr_model_and_diffusion_defaults():
    res = model_and_diffusion_defaults()
    res["large_size"] = 256
    res["small_size"] = 64
    arg_names = inspect.getfullargspec(sr_create_model_and_diffusion)[0]
    for k in res.copy().keys():
        if k not in arg_names:
            del res[k]
    return res


def sr_create_model_and_diffusion(
    large_size,
    small_size,
    class_cond,
    learn_sigma,
    num_channels,
    num_res_blocks,
    num_heads,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
):
    model = sr_create_model(
        large_size,
        small_size,
        num_channels,
        num_res_blocks,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion


def sr_create_model(
    large_size,
    small_size,
    num_channels,
    num_res_blocks,
    learn_sigma,
    class_cond,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
):
    _ = small_size  # hack to prevent unused variable

    if large_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif large_size == 64:
        channel_mult = (1, 2, 3, 4)
    else:
        raise ValueError(f"unsupported large size: {large_size}")

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(large_size // int(res))

    return SuperResModel(
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
    )


def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    b_use_3d_iou=False,
    timestep_respacing="",
    dataset_stats_file: str = None,
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
        if b_use_3d_iou:
            loss_type = gd.LossType.RESCALED_KL_IOU
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
        if b_use_3d_iou:
            loss_type = gd.LossType.RESCALED_MSE_IOU
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]

    used_timestamps = space_timesteps(steps, timestep_respacing)
    # logger.log(f"used_timestamps: {used_timestamps}")
    logger.log(f'dataset_stats_file: {dataset_stats_file}')
    return SpacedDiffusion(
        use_timesteps=used_timestamps,
        betas=betas,
        model_mean_type=(gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X),
        model_var_type=((gd.ModelVarType.FIXED_LARGE if not sigma_small else gd.ModelVarType.FIXED_SMALL)
                        if not learn_sigma else gd.ModelVarType.LEARNED_RANGE),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        dataset_stats_file=dataset_stats_file,
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
