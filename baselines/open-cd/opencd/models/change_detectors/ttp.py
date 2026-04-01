# Copyright (c) Open-CD. All rights reserved.
from typing import List

import torch
from torch import Tensor

from opencd.registry import MODELS
from ..backbones.vit_sam import ViTSAM_Custom  # noqa: F401
from ..backbones.vit_tuner import VisionTransformerTurner  # noqa: F401
from ..utils.ttp_layer import TimeFusionTransformerEncoderLayer  # noqa: F401
from .siamencoder_decoder import SiamEncoderDecoder


@MODELS.register_module()
class TimeTravellingPixels(SiamEncoderDecoder):
    """TTP detector，负责把双时相图像送入共享 ViT-SAM 编码器。"""

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """抽取双时相特征并交给 neck 做时序融合。"""

        img_from, img_to = torch.split(inputs, self.backbone_inchannels, dim=1)
        img = torch.cat([img_from, img_to], dim=0)
        img_feat = self.backbone(img)[0]
        feat_from, feat_to = torch.split(img_feat, img_feat.shape[0] // 2, dim=0)
        feat_from = [feat_from]
        feat_to = [feat_to]
        if self.with_neck:
            x = self.neck(feat_from, feat_to)
        else:
            raise ValueError('`NECK` is needed for `TimeTravellingPixels`.')

        return x
