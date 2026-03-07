import math
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsparse.nn as spnn
from mmengine.registry import MODELS
from torchsparse.nn.utils import fapply
from torchsparse.tensor import SparseTensor

from sam4d.utils import DropPath, get_clones, LayerNorm1d


class SpnnGELU(nn.GELU):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)


class SpnnLayerNorm1d(LayerNorm1d):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)


@MODELS.register_module()
class PtsMemoryEncoder(nn.Module):
    def __init__(
            self,
            out_dim,
            mask_downsampler,
            fuser,
            # position_encoding,
            in_dim=256,  # in_dim of pix_feats
    ):
        super().__init__()

        self.mask_downsampler = MaskDownSampler(**mask_downsampler)

        self.pix_feat_proj = spnn.Conv3d(in_dim, in_dim, kernel_size=1, bias=True)
        self.fuser = Fuser(**fuser)

        self.out_proj = nn.Identity()
        if out_dim != in_dim:
            self.out_proj = spnn.Conv3d(in_dim, out_dim, kernel_size=1, bias=True)

    def forward(
            self,
            pix_feat: torch.Tensor,
            masks: torch.Tensor,
            psam_info: Dict[str, Any],
            skip_mask_sigmoid: bool = False,
    ) -> Dict[str, torch.Tensor]:
        # pix_feat: BxNxC
        # masks: BxCxNx1
        _info = psam_info['pts_sp_tensor_info'][-1]
        pix_feat = [SparseTensor(x, _info['coords'], _info['stride'], _info['spatial_range']).set_caches(_info['_cache']) for x in pix_feat]
        masks = masks.squeeze(3).transpose(1, 2)  # BxNxC
        masks = [SparseTensor(x, psam_info['pts_org_feats'].C.int()) for x in masks]

        ret_x = [self._forward_single(x, m, skip_mask_sigmoid) for x, m in zip(pix_feat, masks)]
        ret_x = torch.stack([x.F for x in ret_x], dim=0).transpose(1, 2).unsqueeze(3)  # BCHW

        return {"vision_features": ret_x}

    def _forward_single(
            self,
            pix_feat: torch.Tensor,
            masks: torch.Tensor,
            skip_mask_sigmoid: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ## Process masks
        # sigmoid, so that less domain shift from gt masks which are bool
        if not skip_mask_sigmoid:
            masks = F.sigmoid(masks)
        masks = self.mask_downsampler(masks)
        masks.F = masks.F.float()  # mask_downsampler has some torch func so it becomes bf16

        ## Fuse pix_feats and downsampled masks in case the visual features are on CPU, cast them to CUDA
        pix_feat = pix_feat.to(masks.device)

        x = self.pix_feat_proj(pix_feat)
        x = x + masks
        x = self.fuser(x)
        x = self.out_proj(x)

        return x


class MaskDownSampler(nn.Module):
    """
    Progressively downsample a mask by total_stride, each time by stride.
    Note that LayerNorm is applied per *token*, like in ViT.

    With each downsample (by a factor stride**2), channel capacity increases by the same factor.
    In the end, we linearly project to embed_dim channels.
    """

    def __init__(
            self,
            embed_dim=256,
            kernel_size=2,
            stride=2,
            padding=0,
            total_stride=16,
    ):
        super().__init__()
        num_layers = int(math.log2(total_stride) // math.log2(stride))
        assert stride ** num_layers == total_stride
        self.encoder = []
        mask_in_chans, mask_out_chans = 1, 1
        for _ in range(num_layers):
            mask_out_chans = mask_in_chans * (stride ** 2)
            self.encoder.append(
                spnn.Conv3d(
                    mask_in_chans,
                    mask_out_chans,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            ),
            # self.encoder.append(spnn.BatchNorm(mask_out_chans))
            # self.encoder.append(spnn.ReLU(inplace=True))
            self.encoder.append(SpnnLayerNorm1d(mask_out_chans))
            self.encoder.append(SpnnGELU())
            mask_in_chans = mask_out_chans

        self.encoder.append(spnn.Conv3d(mask_out_chans, embed_dim, kernel_size=1, bias=True))
        self.encoder = nn.Sequential(*self.encoder)

    def forward(self, x):
        return self.encoder(x)


class Block(nn.Module):
    """
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
            self,
            dim,
            kernel_size=7,
            padding=0,
            drop_path=0.0,
            layer_scale_init_value=1e-6,
            use_dwconv=True,
    ):
        super().__init__()
        self.conv = spnn.Conv3d(dim, dim, kernel_size=kernel_size, padding=padding, )
        self.norm = SpnnLayerNorm1d(dim, eps=1e-6)
        self.pwconv1 = spnn.Conv3d(dim, 4 * dim, kernel_size=1)  # pointwise/1x1 convs, implemented with linear layers
        self.act = SpnnGELU()
        self.pwconv2 = spnn.Conv3d(4 * dim, dim, kernel_size=1)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.conv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x.F = self.gamma * x.F

        x = input + self.drop_path(x)
        return x


class Fuser(nn.Module):
    def __init__(self, layer, num_layers, dim=None, input_projection=False):
        super().__init__()
        self.proj = nn.Identity()
        self.layers = get_clones(Block(**layer), num_layers)

        if input_projection:
            assert dim is not None
            self.proj = spnn.Conv3d(dim, dim, kernel_size=1, bias=True)

    def forward(self, x):
        # normally x: (N, C, H, W)
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x)
        return x
