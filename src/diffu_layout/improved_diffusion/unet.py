from abc import abstractmethod
from typing import Optional

import math
from collections import OrderedDict

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutil

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (SiLU, conv_nd, linear, avg_pool_nd, zero_module, normalization, timestep_embedding, checkpoint,
                 get_activation_fn, FixedPositionalEncoding, LearnedPositionEmbedding)

from . import dist_util, logger
from .attention import SpatialTransformer


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class PositionEmbeddingBlock(nn.Module):
    """
    Any module where forward() takes position embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, pos_emb):
        """
        Apply the module to `x` given `pos_emb` position embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock, PositionEmbeddingBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, ts_emb, context_emb=None, attn_mask=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, ts_emb)
            # elif isinstance(layer, PositionEmbeddingBlock):
            #     x = layer(x, pos_emb, attn_mask)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context_emb, attn_mask)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, channels, channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, channels, channels, 3, stride=stride, padding=1)
        else:
            self.op = avg_pool_nd(stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)

    def _forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        # logger.debug(f"ResBlock::_forward: h.shape: {h.shape}, emb_out.shape: {emb_out.shape}")
        if len(emb_out.shape) == 2:
            emb_out = emb_out[..., None]
        else:
            emb_out = th.permute(emb_out, (0, 2, 1))
        # logger.debug(f'emb_out.shape: {emb_out.shape}')
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class SelfAttnBlock(PositionEmbeddingBlock):

    def __init__(self,
                 in_dim,
                 nhead=8,
                 emb_dim=512,
                 dropout=0.1,
                 activation="relu",
                 use_checkpoint=False,
                 normalize_before=False):
        super().__init__()
        self.num_heads = nhead
        self.self_attn = nn.MultiheadAttention(in_dim, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(in_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(emb_dim, in_dim)

        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(in_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)
        self.use_checkpoint = use_checkpoint
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[th.Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[th.Tensor] = None,
                     src_key_padding_mask: Optional[th.Tensor] = None,
                     pos: Optional[th.Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        logger.debug(f"SelfAttnBlock::forward_post: q.shape: {q.shape}")
        logger.debug(f"SelfAttnBlock::forward_post: k.shape: {k.shape}")
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        logger.debug(f"SelfAttnBlock::forward_post: MHSA.shape: {src2.shape}")
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self,
                    src,
                    src_mask: Optional[th.Tensor] = None,
                    src_key_padding_mask: Optional[th.Tensor] = None,
                    pos: Optional[th.Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self,
                x,
                pos_emb: Optional[th.Tensor] = None,
                attn_mask: Optional[th.Tensor] = None,
                key_padding_mask: Optional[th.Tensor] = None):
        if attn_mask is not None:
            assert attn_mask.shape == (x.shape[0], x.shape[1],
                                       1), f"attn_mask.shape: {attn_mask.shape}, x.shape: {x.shape}"
            attn_mask = (~attn_mask).to(dtype=x.dtype)
            attn_mask = th.matmul(attn_mask, attn_mask.transpose(1, 2)).to(dtype=th.bool)
            attn_mask = ~attn_mask
            attn_mask = attn_mask.repeat(self.num_heads, 1, 1)
            logger.debug(f"SelfAttnBlock::forward: attn_mask.shape: {attn_mask.shape}")
        # else:
        #     attn_mask = th.rand((x.shape[0], x.shape[1], 1)).to(dtype=x.dtype, device=x.device)
        #     attn_mask = (attn_mask > 0.2).to(dtype=x.dtype)
        #     attn_mask = th.matmul(attn_mask, attn_mask.transpose(1, 2)).to(dtype=th.bool)
        #     attn_mask = ~attn_mask
        #     attn_mask = attn_mask.repeat(self.num_heads, 1, 1)

        if self.normalize_before:
            return self.forward_pre(x, attn_mask, key_padding_mask, pos_emb)
        return self.forward_post(x, attn_mask, key_padding_mask, pos_emb)


class AttentionBlock(PositionEmbeddingBlock):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(self, channels, num_heads=1, use_checkpoint=False):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint

        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention()
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x, pos: Optional[th.Tensor] = None, mask: Optional[th.Tensor] = None):
        """ forward function

        Args:
            x (th.Tensor): _description_
            pos (th.Tensor, optional): Positional Embedding. Defaults to None.
            mask (th.Tensor, optional): attention mask. Defaults to None.Bx1xT

        Returns:
            _type_: _description_
        """
        return checkpoint(self._forward, (x, pos, mask), self.parameters(), self.use_checkpoint)

    def _forward(self, x, pos, mask):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        # logger.debug(f"AttentionBlock::forward: qkv.shape: {qkv.shape}")
        if mask is not None:
            # get masks for each head
            mask = mask.repeat(self.num_heads, 1, 1)
        h = self.attention(qkv, pos, mask)
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention.
    """

    def forward(self, qkv, pos: Optional[th.Tensor] = None, attn_mask: Optional[th.Tensor] = None):
        """
        Apply QKV attention.

        :param qkv: an [N x (C * 3) x T] tensor of Qs, Ks, and Vs.
        :param pos: positional embeddings. An [N x C x T] tensor.
        :param attn_mask: an [N x T x T] mask of attention weights.
        :return: an [N x C x T] tensor after attention.
        """
        ch = qkv.shape[1] // 3
        q, k, v = th.split(qkv, ch, dim=1)
        # logger.debug(f"QKVAttention::forward: q.shape: {q.shape}, k.shape: {k.shape}, v.shape: {v.shape}")
        if pos is not None:
            q = q + pos
            k = k + pos
            v = v + pos

        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum("bct,bcs->bts", q * scale, k * scale)  # More stable with f16 than dividing afterwards
        # logger.debug(f"QKVAttention::forward: weight.shape: {weight.shape}")
        # logger.debug(f"QKVAttention::forward: attn_mask[0,-1,:]: {attn_mask[0,-1,:]}")
        if attn_mask is not None:
            weight = weight.masked_fill(attn_mask, -1e9)
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        return th.einsum("bts,bcs->bct", weight, v)

    @staticmethod
    def count_flops(model, _x, y):
        """
        A counter for the `thop` package to count the operations in an
        attention operation.

        Meant to be used like:

            macs, params = thop.profile(
                model,
                inputs=(inputs, timestamps),
                custom_ops={QKVAttention: QKVAttention.count_flops},
            )

        """
        b, c, *spatial = y[0].shape
        num_spatial = int(np.prod(spatial))
        # We perform two matmuls with the same number of ops.
        # The first computes the weight matrix, the second computes
        # the combination of the value vectors.
        matmul_ops = 2 * b * (num_spatial**2) * c
        model.total_ops += th.DoubleTensor([matmul_ops])


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param text_emb_dim: dimension of input text embedding.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param use_input_encoding: if True, use an encoding network before the UNet
    :param class_label_feat_size: size of the class label embedding.
    :param bbox_center_feat_size: size of the bounding box center embedding.
    :param bbox_size_feat_size: size of the bounding box size embedding.
    :param bbox_angle_feat_size: size of the bounding box angle embedding.
    :param attn_block_depth: depth of the attention block.
    """

    def __init__(
        self,
        in_channels,  # input feature channels
        model_channels,  # basic channel number of the model
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=1,
        text_condition=True,  # whether use text as condition
        text_emb_dim=768,  # text embedding dimension (77,768) for BERT
        use_checkpoint=False,
        num_heads=1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        use_input_encoding=False,
        # input data properties' indices
        class_label_feat_size=32,
        bbox_center_feat_size=32,
        bbox_size_feat_size=32,
        bbox_angle_feat_size=32,
        # attention blocks
        attn_block_depth=1,
        num_instances: int = 21,  # number of instances
        instance_emb_dim: int = 512,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.use_input_encoding = use_input_encoding
        self.object_emb_dim = in_channels

        # encode input data to encoding space,
        if self.use_input_encoding:
            # Embedding matix for property class label.
            # Compute the number of classes from the input_dims. Note that we
            # remove 3 to account for the masked bins for the size, 3 for position and
            # 2 for angle properties
            self.num_object_classes = in_channels - 3 - 3 - 2
            # self.object_class_emb_layer = nn.Linear(self.num_object_classes, class_label_feat_size, bias=False)
            # # Positional encoding for each property
            # self.object_pe_pos_x = FixedPositionalEncoding(proj_dims=bbox_center_feat_size)
            # self.object_pe_pos_y = FixedPositionalEncoding(proj_dims=bbox_center_feat_size)
            # self.object_pe_pos_z = FixedPositionalEncoding(proj_dims=bbox_center_feat_size)
            # self.object_pe_size_x = FixedPositionalEncoding(proj_dims=bbox_size_feat_size)
            # self.object_pe_size_y = FixedPositionalEncoding(proj_dims=bbox_size_feat_size)
            # self.object_pe_size_z = FixedPositionalEncoding(proj_dims=bbox_size_feat_size)
            # self.object_pe_angle_cz = FixedPositionalEncoding(proj_dims=bbox_angle_feat_size)
            # self.object_pe_angle_sz = FixedPositionalEncoding(proj_dims=bbox_angle_feat_size)
            # self.object_emb_dim = class_label_feat_size + bbox_center_feat_size * 3 + bbox_size_feat_size * 3 + bbox_angle_feat_size * 2

            self.class_embed_func = UNetModel._encoder_mlp(hidden_size=model_channels,
                                                           input_size=self.num_object_classes)
            self.bbox_embed_func = UNetModel._encoder_mlp(hidden_size=model_channels, input_size=8)
            self.object_emb_dim = model_channels * 2

        self.in_channels = self.object_emb_dim
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.text_emb_dim = text_emb_dim
        self.b_text_cond = text_condition
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample

        # attention blocks
        self.attn_block_depth = attn_block_depth
        self.num_instances = num_instances
        self.instance_emb_dim = instance_emb_dim

        # time embedding block
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        # instance embedding block
        if self.num_instances is not None:
            self.instance_embed_layer = nn.Embedding(num_instances, instance_emb_dim)
        else:
            self.instance_embed_layer = None

        # initial block
        self.init_conv = conv_nd(dims, self.in_channels, model_channels, 1)  # 3, padding=1
        # input block
        self.input_blocks = nn.ModuleList([])
        input_block_chans = [model_channels]
        ch = model_channels

        for level, mult in enumerate(channel_mult):
            for idx in range(num_res_blocks):
                idx_str = f'{level}_{idx}'
                layers = [(
                    'resblock_' + idx_str + '_0',
                    ResBlock(
                        channels=ch,  # input channels
                        emb_channels=self.instance_emb_dim,  # embedding channels
                        dropout=dropout,  # dropout probability
                        out_channels=ch,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    ))]
                layers.append((
                    'resblock_' + idx_str + '_1',
                    ResBlock(
                        channels=ch,  # input channels
                        emb_channels=time_embed_dim,  # embedding channels
                        dropout=dropout,  # dropout probability
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )))
                # increase minput channels for next resblock
                ch = mult * model_channels
                # channels for q.k.v of attention block
                dim_heads = ch // num_heads

                layers.append(('attnblock_' + idx_str,
                               SpatialTransformer(in_channels=ch,
                                                  n_heads=num_heads,
                                                  d_head=dim_heads,
                                                  depth=self.attn_block_depth,
                                                  dropout=0.1,
                                                  context_dim=self.text_emb_dim,
                                                  disable_self_attn=False,
                                                  use_linear=False,
                                                  use_checkpoint=use_checkpoint,
                                                  dims=dims)))

                layers.append(('down_' + idx_str, conv_nd(dims, ch, ch, 1)))  # 3, padding=1
                self.input_blocks.append(TimestepEmbedSequential(OrderedDict(layers)))
                input_block_chans.append(ch)
            # if level != len(channel_mult) - 1:
            #     self.input_blocks.append(
            #         TimestepEmbedSequential(OrderedDict([('down_' + idx_str, conv_nd(dims, ch, ch, 3, padding=1))])))
            #     input_block_chans.append(ch)

        # out_channel is same as input_channel in moddle block
        self.middle_block = nn.ModuleList([])
        self.middle_block.append(
            TimestepEmbedSequential(
                ResBlock(
                    ch,
                    self.instance_emb_dim,
                    dropout,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                ),
                ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                ),
                SpatialTransformer(in_channels=ch,
                                   n_heads=num_heads,
                                   d_head=ch // num_heads,
                                   depth=self.attn_block_depth,
                                   dropout=0.1,
                                   context_dim=self.text_emb_dim,
                                   disable_self_attn=False,
                                   use_linear=False,
                                   use_checkpoint=use_checkpoint,
                                   dims=dims),
                ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                ),
            ))

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            # for i in range(num_res_blocks + 1):
            for i in range(num_res_blocks):
                idx_str = f'{level}_{i}'
                layers = [('resblock_' + idx_str + '_0',
                           ResBlock(
                               channels=ch,
                               emb_channels=self.instance_emb_dim,
                               dropout=dropout,
                               out_channels=ch,
                               dims=dims,
                               use_checkpoint=use_checkpoint,
                               use_scale_shift_norm=use_scale_shift_norm,
                           ))]
                layers.append(('resblock_' + idx_str + '_1',
                               ResBlock(
                                   channels=ch + input_block_chans.pop(),
                                   emb_channels=time_embed_dim,
                                   dropout=dropout,
                                   out_channels=model_channels * mult,
                                   dims=dims,
                                   use_checkpoint=use_checkpoint,
                                   use_scale_shift_norm=use_scale_shift_norm,
                               )))
                ch = model_channels * mult
                # channels for q.k.v of attention block
                dim_heads = ch // num_heads

                layers.append(('attnblock_' + idx_str,
                               SpatialTransformer(in_channels=ch,
                                                  n_heads=num_heads_upsample,
                                                  d_head=dim_heads,
                                                  depth=self.attn_block_depth,
                                                  dropout=0.1,
                                                  context_dim=self.text_emb_dim,
                                                  disable_self_attn=False,
                                                  use_linear=False,
                                                  use_checkpoint=use_checkpoint,
                                                  dims=dims)))

                layers.append(('up_' + idx_str, conv_nd(dims, ch, ch, 1)))  # 3, padding=1
                # if level and i == num_res_blocks:
                #     # layers.append(('up_' + idx_str, Upsample(channels=ch, use_conv=conv_resample, dims=dims)))
                #     layers.append(('conv_' + idx_str, conv_nd(dims, ch, ch, 3, padding=1)))
                self.output_blocks.append(TimestepEmbedSequential(OrderedDict(layers)))

        self.final_res_block = ResBlock(
            channels=ch,
            emb_channels=time_embed_dim,
            dropout=dropout,
            out_channels=ch,
            dims=dims,
            use_checkpoint=use_checkpoint,
            use_scale_shift_norm=use_scale_shift_norm,
        )

        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 1)),  # 3, padding=1
        )

        # as we encode input data to encoding space, we need to project it back to input space
        # if self.use_input_encoding:
        #     self.proj_out = nn.Sequential(OrderedDict([('project_out', nn.Linear(self.object_emb_dim, in_channels))]))
        #     self.proj_out.register_forward_hook(self._proj_out_hook_func)

    def _proj_out_hook_func(self, module, input, output):
        """ hook function of proj_out layer, visualize the output of proj_out layer """
        data = output.clone().detach()
        logger.debug(f"_proj_out_hook_func: {data.shape}")
        # data = data.unsqueeze(0)
        # data = data.permute(1, 0, 2, 3)
        # logger.debug(f"_proj_out_hook_func: {data.shape}")
        # img_name = "proj_out_tensor.png"
        # vutil.save_image(data, img_name, pad_value=0.5)
        # grid_img = vutil.make_grid(data, nrow=1, normalize=True, scale_each=True, pad_value=0.5)
        # vutil.save_image(grid_img, img_name, pad_value=0.5)

    @staticmethod
    def _encoder_mlp(hidden_size, input_size):
        mlp_layers = [
            nn.Conv1d(input_size, hidden_size, 1),
            nn.GELU(),
            nn.Conv1d(hidden_size, hidden_size * 2, 1),
            nn.GELU(),
            nn.Conv1d(hidden_size * 2, hidden_size, 1),
        ]
        return nn.Sequential(*mlp_layers)

    @staticmethod
    def _decoder_mlp(hidden_size, output_size):
        mlp_layers = [
            nn.Conv1d(hidden_size, hidden_size * 2, 1),
            nn.GELU(),
            nn.Conv1d(hidden_size * 2, hidden_size, 1),
            nn.GELU(),
            nn.Conv1d(hidden_size, output_size, 1),
        ]
        return nn.Sequential(*mlp_layers)

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype

    def forward(
        self,
        x,
        timesteps,
        # instance_condition=None,
        text_condition=None,
    ):
        """
        Apply the model to an input batch.
        :param x: an [B x C x N] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param instance_condition: an [B x N] Tensor of instances, if instance-conditional.
        :param text_condition: conditioning plugged in via crossattn

        :return: an [B x C x 2N] Tensor of outputs.
        """
        # if self.num_instances is not None:
        #     assert instance_condition is not None, "must specify instance_condition if and only if the model is instance-conditional"

        hs = []
        logger.debug(f"UNetModel::forward: input x: {x.shape}")
        # logger.debug(f"UNetModel::forward: input timesteps shape: {timesteps.shape}")
        time_emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        # logger.debug(f"UNetModel::forward: output timesteps embedding shape: {time_emb.shape}")

        X = x
        if self.use_input_encoding:
            # x = x.transpose(1, 2)
            # object_class_labels = x[:, :, :self.num_object_classes]
            # object_bbox_poses = x[:, :, self.num_object_classes:self.num_object_classes + 3]
            # object_bbox_sizes = x[:, :, self.num_object_classes + 3:self.num_object_classes + 6]
            # object_bbox_angles = x[:, :, self.num_object_classes + 6:self.num_object_classes + 8]
            # object_class_feat = self.object_class_emb_layer(object_class_labels)
            # # logger.debug(f"UNetModel::forward: output object class embedding shape: {object_class_feat.shape}")
            # object_pos_x_feat = self.object_pe_pos_x(object_bbox_poses[:, :, 0:1])
            # object_pos_y_feat = self.object_pe_pos_y(object_bbox_poses[:, :, 1:2])
            # object_pos_z_feat = self.object_pe_pos_z(object_bbox_poses[:, :, 2:3])
            # object_pos_feat = th.cat([object_pos_x_feat, object_pos_y_feat, object_pos_z_feat], dim=-1)
            # # logger.debug(f"UNetModel::forward: output object position embedding shape: {object_pos_feat.shape}")
            # object_size_x_feat = self.object_pe_size_x(object_bbox_sizes[:, :, 0:1])
            # object_size_y_feat = self.object_pe_size_y(object_bbox_sizes[:, :, 1:2])
            # object_size_z_feat = self.object_pe_size_z(object_bbox_sizes[:, :, 2:3])
            # object_size_feat = th.cat([object_size_x_feat, object_size_y_feat, object_size_z_feat], dim=-1)
            # # logger.debug(f"UNetModel::forward: output object size embedding shape: {object_size_feat.shape}")
            # object_angle_x_feat = self.object_pe_angle_cz(object_bbox_angles[:, :, 0:1])
            # object_angle_y_feat = self.object_pe_angle_sz(object_bbox_angles[:, :, 1:2])
            # object_angle_feat = th.cat([object_angle_x_feat, object_angle_y_feat], dim=-1)
            # # logger.debug(f"UNetModel::forward: output object angle embedding shape: {object_angle_feat.shape}")

            # X = th.cat([object_class_feat, object_pos_feat, object_size_feat, object_angle_feat], dim=-1)
            # X = X.transpose(1, 2)

            object_class_feat = self.class_embed_func(x[:, :self.num_object_classes, :])
            logger.debug(f"UNetModel::forward: output object class embedding shape: {object_class_feat.shape}")
            object_bbox_feat = self.bbox_embed_func(x[:, self.num_object_classes:self.num_object_classes + 8, :])
            logger.debug(f"UNetModel::forward: output object bbox embedding shape: {object_bbox_feat.shape}")
            X = th.cat([object_class_feat, object_bbox_feat], dim=1)

        logger.debug(f"UNetModel::forward:  X shape: {X.shape}")

        context_emb = None
        # if self.num_classes is not None:
        #     assert y.shape == (X.shape[0],)
        #     time_emb = time_emb + self.label_emb(y)
        # logger.debug(f"UNetModel::forward: input y shape: {y.shape}")
        instance_emb = None
        if self.instance_embed_layer is not None:
            instance_indices = th.arange(self.num_instances).long().to(x.device)[None, :].repeat(x.shape[0], 1)
            instance_emb = self.instance_embed_layer(instance_indices)
            logger.debug(f"UNetModel::forward: instance_emb.shape: {instance_emb.shape}")
            # time_emb = time_emb + instance_emb
        if self.b_text_cond and text_condition is not None:
            assert text_condition.shape == (
                X.shape[0], 77,
                self.text_emb_dim), f" expected input text_condition.shape: {(X.shape[0],77, self.text_emb_dim)}"
            context_emb = text_condition
            while len(context_emb.shape) < len(X.shape):
                context_emb = context_emb.unsqueeze(1)
            # logger.debug(f"UNetModel::forward: context_emb.shape: {context_emb.shape}")
        else:
            raise ValueError("Must specify either instance_condition or text_condition")

        h = X.type(self.inner_dtype)

        h = self.init_conv(h)
        # down blocks
        for (res0, res1, attn0, dsm) in self.input_blocks:
            h = res0(h, instance_emb)
            h = res1(h, time_emb)
            h = attn0(h, context_emb)
            h = dsm(h)
            hs.append(h)

        # middle blocks
        for (res0, res1, attn0, res2) in self.middle_block:
            h = res0(h, instance_emb)
            h = res1(h, time_emb)
            h = attn0(h, context_emb)
            h = res2(h, time_emb)

        # up blocks
        for (res0, res1, attn0, usm) in self.output_blocks:
            h = res0(h, instance_emb)
            h_in_input = hs.pop()
            cat_in = th.cat([h, h_in_input], dim=1)
            h = res1(cat_in, time_emb)
            h = attn0(h, context_emb)
            h = usm(h)

        h = self.final_res_block(h, time_emb)
        h = h.type(X.dtype)
        ret = self.out(h)
        # logger.debug(f"UNetModel::forward: out layer output shape: {ret.shape}")

        return ret

    def get_feature_vectors(self, x, timesteps, y=None):
        """
        Apply the model and return all of the intermediate tensors.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: a dict with the following keys:
                 - 'down': a list of hidden state tensors from downsampling.
                 - 'middle': the tensor of the output of the lowest-resolution
                             block in the model.
                 - 'up': a list of hidden state tensors from upsampling.
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        result = dict(down=[], up=[])
        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
            result["down"].append(h.type(x.dtype))
        h = self.middle_block(h, emb)
        result["middle"] = h.type(x.dtype)
        for module in self.output_blocks:
            cat_in = th.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
            result["up"].append(h.type(x.dtype))
        return result


class SuperResModel(UNetModel):
    """
    A UNetModel that performs super-resolution.

    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, in_channels, *args, **kwargs):
        super().__init__(in_channels * 2, *args, **kwargs)

    def forward(self, x, timesteps, low_res=None, **kwargs):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().forward(x, timesteps, **kwargs)

    def get_feature_vectors(self, x, timesteps, low_res=None, **kwargs):
        _, new_height, new_width, _ = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().get_feature_vectors(x, timesteps, **kwargs)
