# Copyright (c) Open-CD. All rights reserved.
import collections.abc
from typing import Optional, Sequence, Tuple
from itertools import repeat

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import FFN, PatchEmbed
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import trunc_normal_

from opencd.registry import MODELS


def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return tuple(repeat(x, 2))


def resize_pos_embed(pos_embed,
                     src_shape,
                     dst_shape,
                     mode='bicubic',
                     num_extra_tokens=1):
    """Resize pos_embed 权重。"""

    if src_shape[0] == dst_shape[0] and src_shape[1] == dst_shape[1]:
        return pos_embed
    assert pos_embed.ndim == 3, 'shape of pos_embed must be [1, L, C]'
    _, l, c = pos_embed.shape
    src_h, src_w = src_shape
    assert l == src_h * src_w + num_extra_tokens, (
        f"The length of `pos_embed` ({l}) doesn't match the expected "
        f'shape ({src_h}*{src_w}+{num_extra_tokens}). Please check the'
        '`img_size` argument.')
    extra_tokens = pos_embed[:, :num_extra_tokens]

    src_weight = pos_embed[:, num_extra_tokens:]
    src_weight = src_weight.reshape(1, src_h, src_w, c).permute(0, 3, 1, 2)

    dst_weight = F.interpolate(
        src_weight.float(), size=dst_shape, align_corners=False, mode=mode)
    dst_weight = torch.flatten(dst_weight, 2).transpose(1, 2)
    dst_weight = dst_weight.to(src_weight.dtype)

    return torch.cat((extra_tokens, dst_weight), dim=1)


@MODELS.register_module(name='LN2d')
class LayerNorm2d(nn.LayerNorm):
    """2D feature map 上按通道做 LayerNorm。"""

    def __init__(self, num_channels: int, **kwargs) -> None:
        super().__init__(num_channels, **kwargs)
        self.num_channels = self.normalized_shape[0]

    def forward(self, x, data_format='channel_first'):
        assert x.dim() == 4, 'LayerNorm2d only supports inputs with shape ' \
            f'(N, C, H, W), but got tensor with shape {x.shape}'
        if data_format == 'channel_last':
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias,
                             self.eps)
        elif data_format == 'channel_first':
            x = x.permute(0, 2, 3, 1)
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias,
                             self.eps)
            x = x.permute(0, 3, 1, 2).contiguous()
        return x


def build_norm_layer(cfg: dict, num_features: int) -> nn.Module:
    """为 TTP 本地化最小 norm 构造逻辑，避免依赖 mmpretrain registry。"""

    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-5)

    if layer_type == 'LN':
        layer = nn.LayerNorm(num_features, **cfg_)
    elif layer_type in {'LN2d', 'LayerNorm2d'}:
        layer = LayerNorm2d(num_features, **cfg_)
    elif layer_type == 'SyncBN':
        layer = nn.SyncBatchNorm(num_features, **cfg_)
        if hasattr(layer, '_specify_ddp_gpu_num'):
            layer._specify_ddp_gpu_num(1)
    elif layer_type in {'BN', 'BN2d', 'BatchNorm2d'}:
        layer = nn.BatchNorm2d(num_features, **cfg_)
    else:
        raise KeyError(f'Unsupported norm layer type: {layer_type}')

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return layer


def window_partition(x: torch.Tensor,
                     window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """将特征划分为窗口并在必要时做 padding。"""

    b, h, w, c = x.shape

    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    hp, wp = h + pad_h, w + pad_w

    x = x.view(b, hp // window_size, window_size, wp // window_size,
               window_size, c)
    windows = x.permute(0, 1, 3, 2, 4,
                        5).contiguous().view(-1, window_size, window_size, c)
    return windows, (hp, wp)


def window_unpartition(windows: torch.Tensor, window_size: int,
                       pad_hw: Tuple[int, int],
                       hw: Tuple[int, int]) -> torch.Tensor:
    """把窗口特征还原回原始特征图。"""

    hp, wp = pad_hw
    h, w = hw
    b = windows.shape[0] // (hp * wp // window_size // window_size)
    x = windows.view(b, hp // window_size, wp // window_size, window_size,
                     window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, hp, wp, -1)

    if hp > h or wp > w:
        x = x[:, :h, :w, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int,
                rel_pos: torch.Tensor) -> torch.Tensor:
    """根据 query/key 大小获取相对位置编码。"""

    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode='linear',
        )
        rel_pos_resized = rel_pos_resized.reshape(-1,
                                                  max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords -
                       k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(attn: torch.Tensor, q: torch.Tensor,
                           rel_pos_h: torch.Tensor, rel_pos_w: torch.Tensor,
                           q_size: Tuple[int, int],
                           k_size: Tuple[int, int]) -> torch.Tensor:
    """给注意力图加入分解式相对位置编码。"""

    q_h, q_w = q_size
    k_h, k_w = k_size
    rh = get_rel_pos(q_h, k_h, rel_pos_h)
    rw = get_rel_pos(q_w, k_w, rel_pos_w)

    b, _, dim = q.shape
    r_q = q.reshape(b, q_h, q_w, dim)
    rel_h = torch.einsum('bhwc,hkc->bhwk', r_q, rh)
    rel_w = torch.einsum('bhwc,wkc->bhwk', r_q, rw)

    attn = (attn.view(b, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] +
            rel_w[:, :, :, None, :]).view(b, q_h * q_w, k_h * k_w)

    return attn


class Attention(nn.Module):
    """SAM 风格的多头注意力。"""

    def __init__(self,
                 embed_dims: int,
                 num_heads: int = 8,
                 qkv_bias: bool = True,
                 use_rel_pos: bool = False,
                 input_size: Optional[Tuple[int, int]] = None) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = head_embed_dims**-0.5

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.proj = nn.Linear(embed_dims, embed_dims)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None, (
                'Input size must be provided if using relative position embed.')
            self.rel_pos_h = nn.Parameter(
                torch.zeros(2 * input_size[0] - 1, head_embed_dims))
            self.rel_pos_w = nn.Parameter(
                torch.zeros(2 * input_size[1] - 1, head_embed_dims))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, h, w, _ = x.shape
        qkv = self.qkv(x).reshape(b, h * w, 3, self.num_heads,
                                  -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, b * self.num_heads, h * w, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h,
                                          self.rel_pos_w, (h, w), (h, w))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(b, self.num_heads, h, w,
                            -1).permute(0, 2, 3, 1, 4).reshape(b, h, w, -1)
        x = self.proj(x)
        return x


class BaseBackbone(BaseModule):
    """最小骨干基类，避免再依赖 mmpretrain 的 backbone 抽象。"""

    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)


class TransformerEncoderLayer(BaseModule):
    """本地化的 SAM 编码层实现。"""

    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 feedforward_channels: int,
                 drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 num_fcs: int = 2,
                 qkv_bias: bool = True,
                 act_cfg: dict = dict(type='GELU'),
                 norm_cfg: dict = dict(type='LN'),
                 use_rel_pos: bool = False,
                 window_size: int = 0,
                 input_size: Optional[Tuple[int, int]] = None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        self.window_size = window_size

        self.ln1 = build_norm_layer(norm_cfg, self.embed_dims)

        self.attn = Attention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            input_size=input_size if window_size == 0 else
            (window_size, window_size),
        )

        self.ln2 = build_norm_layer(norm_cfg, self.embed_dims)

        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=num_fcs,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg)

    @property
    def norm1(self):
        return self.ln1

    @property
    def norm2(self):
        return self.ln2

    def forward(self, x):
        shortcut = x
        x = self.ln1(x)
        if self.window_size > 0:
            h, w = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (h, w))
        x = shortcut + x

        x = self.ffn(self.ln2(x), identity=x)
        return x


@MODELS.register_module()
class ViTSAM_Custom(BaseBackbone):
    """Vision Transformer as image encoder used in SAM.

    A PyTorch implement of backbone: `Segment Anything
    <https://arxiv.org/abs/2304.02643>`_

    Args:
        arch (str | dict): Vision Transformer architecture. If use string,
            choose from 'base', 'large', 'huge'. If use dict, it should have
            below keys:

            - **embed_dims** (int): The dimensions of embedding.
            - **num_layers** (int): The number of transformer encoder layers.
            - **num_heads** (int): The number of heads in attention modules.
            - **feedforward_channels** (int): The hidden dimensions in
              feedforward modules.
            - **global_attn_indexes** (int): The index of layers with global
              attention.

            Defaults to 'base'.
        img_size (int | tuple): The expected input image shape. Because we
            support dynamic input shape, just set the argument to the most
            common input image shape. Defaults to 224.
        patch_size (int | tuple): The patch size in patch embedding.
            Defaults to 16.
        in_channels (int): The num of input channels. Defaults to 3.
        out_channels (int): The num of output channels, if equal to 0, the
            channel reduction layer is disabled. Defaults to 256.
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        out_type (str): The type of output features. Please choose from

            - ``"raw"`` or ``"featmap"``: The feature map tensor from the
              patch tokens with shape (B, C, H, W).
            - ``"avg_featmap"``: The global averaged feature map tensor
              with shape (B, C).

            Defaults to ``"raw"``.
        drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        qkv_bias (bool): Whether to add bias for qkv in attention modules.
            Defaults to True.
        use_abs_pos (bool): Whether to use absolute position embedding.
            Defaults to True.
        use_rel_pos (bool):Whether to use relative position embedding.
            Defaults to True.
        window_size (int): Window size for window attention. Defaults to 14.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Defaults to -1.
        interpolate_mode (str): Select the interpolate mode for position
            embeding vector resize. Defaults to "bicubic".
        patch_cfg (dict): Configs of patch embeding. Defaults to an empty dict.
        layer_cfgs (Sequence | dict): Configs of each transformer layer in
            encoder. Defaults to an empty dict.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """
    arch_zoo = {
        **dict.fromkeys(
            ['b', 'base'], {
                'embed_dims': 768,
                'num_layers': 12,
                'num_heads': 12,
                'feedforward_channels': 3072,
                'global_attn_indexes': [2, 5, 8, 11]
            }),
        **dict.fromkeys(
            ['l', 'large'], {
                'embed_dims': 1024,
                'num_layers': 24,
                'num_heads': 16,
                'feedforward_channels': 4096,
                'global_attn_indexes': [5, 11, 17, 23]
            }),
        **dict.fromkeys(
            ['h', 'huge'], {
                'embed_dims': 1280,
                'num_layers': 32,
                'num_heads': 16,
                'feedforward_channels': 5120,
                'global_attn_indexes': [7, 15, 23, 31]
            }),
    }
    OUT_TYPES = {'raw', 'featmap', 'avg_featmap'}

    def __init__(self,
                 arch: str = 'base',
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 out_channels: int = 256,
                 out_indices: int = -1,
                 out_type: str = 'raw',
                 drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 qkv_bias: bool = True,
                 use_abs_pos: bool = True,
                 use_rel_pos: bool = True,
                 window_size: int = 14,
                 norm_cfg: dict = dict(type='LN', eps=1e-6),
                 frozen_stages: int = -1,
                 interpolate_mode: str = 'bicubic',
                 patch_cfg: dict = dict(),
                 layer_cfgs: dict = dict(),
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg)

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {
                'embed_dims', 'num_layers', 'num_heads', 'feedforward_channels'
            }
            assert isinstance(arch, dict) and essential_keys <= set(arch), \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        self.embed_dims = self.arch_settings['embed_dims']
        self.num_layers = self.arch_settings['num_layers']
        self.global_attn_indexes = self.arch_settings['global_attn_indexes']
        self.img_size = to_2tuple(img_size)

        # Set patch embedding
        _patch_cfg = dict(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
        )
        _patch_cfg.update(patch_cfg)
        self.patch_embed = PatchEmbed(**_patch_cfg)
        self.patch_resolution = self.patch_embed.init_out_size

        # Set out type
        if out_type not in self.OUT_TYPES:
            raise ValueError(f'Unsupported `out_type` {out_type}, please '
                             f'choose from {self.OUT_TYPES}')
        self.out_type = out_type

        self.use_abs_pos = use_abs_pos
        self.interpolate_mode = interpolate_mode
        if use_abs_pos:
            # Set position embedding
            self.pos_embed = nn.Parameter(
                torch.zeros(1, *self.patch_resolution, self.embed_dims))
            self.drop_after_pos = nn.Dropout(p=drop_rate)
            self._register_load_state_dict_pre_hook(self._prepare_pos_embed)

        if use_rel_pos:
            self._register_load_state_dict_pre_hook(
                self._prepare_relative_position)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_layers + index
            assert 0 <= out_indices[i] <= self.num_layers, \
                f'Invalid out_indices {index}'
        self.out_indices = out_indices

        # stochastic depth decay rule
        dpr = np.linspace(0, drop_path_rate, self.num_layers)

        self.layers = ModuleList()
        if isinstance(layer_cfgs, dict):
            layer_cfgs = [layer_cfgs] * self.num_layers
        for i in range(self.num_layers):
            _layer_cfg = dict(
                embed_dims=self.embed_dims,
                num_heads=self.arch_settings['num_heads'],
                feedforward_channels=self.
                arch_settings['feedforward_channels'],
                drop_rate=drop_rate,
                drop_path_rate=dpr[i],
                qkv_bias=qkv_bias,
                window_size=window_size
                if i not in self.global_attn_indexes else 0,
                input_size=self.patch_resolution,
                use_rel_pos=use_rel_pos,
                norm_cfg=norm_cfg)
            _layer_cfg.update(layer_cfgs[i])
            if 'type' in _layer_cfg:
                self.layers.append(MODELS.build(_layer_cfg))
            else:
                self.layers.append(TransformerEncoderLayer(**_layer_cfg))

        self.out_channels = out_channels
        if self.out_channels > 0:
            self.channel_reduction = nn.Sequential(
                nn.Conv2d(
                    self.embed_dims,
                    out_channels,
                    kernel_size=1,
                    bias=False,
                ),
                LayerNorm2d(out_channels, eps=1e-6),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                LayerNorm2d(out_channels, eps=1e-6),
            )

        # freeze stages only when self.frozen_stages > 0
        self.frozen_stages = frozen_stages
        if self.frozen_stages > 0:
            self._freeze_stages()

    def init_weights(self):
        super().init_weights()

        if not (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            if self.pos_embed is not None:
                trunc_normal_(self.pos_embed, std=0.02)

    def _freeze_stages(self):
        # freeze position embedding
        if self.pos_embed is not None:
            self.pos_embed.requires_grad = False
        # set dropout to eval model
        self.drop_after_pos.eval()
        # freeze patch embedding
        self.patch_embed.eval()
        for param in self.patch_embed.parameters():
            param.requires_grad = False

        # freeze layers
        for i in range(1, self.frozen_stages + 1):
            m = self.layers[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

        # freeze channel_reduction module
        if self.frozen_stages == self.num_layers and self.out_channels > 0:
            m = self.channel_reduction
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        B = x.shape[0]
        x, patch_resolution = self.patch_embed(x)
        x = x.view(B, patch_resolution[0], patch_resolution[1],
                   self.embed_dims)

        if self.use_abs_pos:
            # 'resize_pos_embed' only supports 'pos_embed' with ndim==3, but
            # in ViTSAM, the 'pos_embed' has 4 dimensions (1, H, W, C), so it
            # is flattened. Besides, ViTSAM doesn't have any extra token.
            resized_pos_embed = resize_pos_embed(
                self.pos_embed.flatten(1, 2),
                self.patch_resolution,
                patch_resolution,
                mode=self.interpolate_mode,
                num_extra_tokens=0)
            x = x + resized_pos_embed.view(1, *patch_resolution,
                                           self.embed_dims)
            x = self.drop_after_pos(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i in self.out_indices:
                # (B, H, W, C) -> (B, C, H, W)
                x_reshape = x.permute(0, 3, 1, 2)

                if self.out_channels > 0:
                    x_reshape = self.channel_reduction(x_reshape)
                outs.append(self._format_output(x_reshape))

        return tuple(outs)

    def _format_output(self, x) -> torch.Tensor:
        if self.out_type == 'raw' or self.out_type == 'featmap':
            return x
        elif self.out_type == 'avg_featmap':
            # (B, C, H, W) -> (B, C, N) -> (B, N, C)
            x = x.flatten(2).permute(0, 2, 1)
            return x.mean(dim=1)

    def _prepare_pos_embed(self, state_dict, prefix, *args, **kwargs):
        name = prefix + 'pos_embed'
        if name not in state_dict.keys():
            return

        ckpt_pos_embed_shape = state_dict[name].shape
        if self.pos_embed.shape != ckpt_pos_embed_shape:
            from mmengine.logging import MMLogger
            logger = MMLogger.get_current_instance()
            logger.info(
                f'Resize the pos_embed shape from {ckpt_pos_embed_shape} '
                f'to {self.pos_embed.shape}.')

            ckpt_pos_embed_shape = ckpt_pos_embed_shape[1:3]
            pos_embed_shape = self.patch_embed.init_out_size

            flattened_pos_embed = state_dict[name].flatten(1, 2)
            resized_pos_embed = resize_pos_embed(flattened_pos_embed,
                                                 ckpt_pos_embed_shape,
                                                 pos_embed_shape,
                                                 self.interpolate_mode, 0)
            state_dict[name] = resized_pos_embed.view(1, *pos_embed_shape,
                                                      self.embed_dims)

    def _prepare_relative_position(self, state_dict, prefix, *args, **kwargs):
        state_dict_model = self.state_dict()
        all_keys = list(state_dict_model.keys())
        for key in all_keys:
            if 'rel_pos_' in key:
                ckpt_key = prefix + key
                if ckpt_key not in state_dict:
                    continue
                relative_position_pretrained = state_dict[ckpt_key]
                relative_position_current = state_dict_model[key]
                L1, _ = relative_position_pretrained.size()
                L2, _ = relative_position_current.size()
                if L1 != L2:
                    new_rel_pos = F.interpolate(
                        relative_position_pretrained.reshape(1, L1,
                                                             -1).permute(
                                                                 0, 2, 1),
                        size=L2,
                        mode='linear',
                    )
                    new_rel_pos = new_rel_pos.reshape(-1, L2).permute(1, 0)
                    from mmengine.logging import MMLogger
                    logger = MMLogger.get_current_instance()
                    logger.info(f'Resize the {ckpt_key} from '
                                f'{state_dict[ckpt_key].shape} to '
                                f'{new_rel_pos.shape}')
                    state_dict[ckpt_key] = new_rel_pos

    def get_layer_depth(self, param_name: str, prefix: str = ''):
        """Get the layer-wise depth of a parameter.

        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.

        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.

        Note:
            The first depth is the stem module (``layer_depth=0``), and the
            last depth is the subsequent module (``layer_depth=num_layers-1``)
        """
        num_layers = self.num_layers + 2

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return num_layers - 1, num_layers

        param_name = param_name[len(prefix):]

        if param_name in ('cls_token', 'pos_embed'):
            layer_depth = 0
        elif param_name.startswith('patch_embed'):
            layer_depth = 0
        elif param_name.startswith('layers'):
            layer_id = int(param_name.split('.')[1])
            layer_depth = layer_id + 1
        else:
            layer_depth = num_layers - 1

        return layer_depth, num_layers
