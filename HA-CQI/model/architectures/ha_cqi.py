import torch
import torch.nn as nn

from ..decode_heads import Mask2FormerChangeHead, MultiScaleAuxiliaryHead
from ..modules import ChangeQueryInteraction, HarmonizedAlignment, HierarchicalCnnDinoEncoder


class HACQIModel(nn.Module):
    """HA-CQI 主模型：Encoder -> HA -> CQI -> Mask2Former change head。"""

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        fpn_channels: int = 128,
        n_layers=None,
        disable_soft_alignment: bool = False,
        align_window: int = 5,
        align_points: int = 9,
        align_heads: int = 4,
        align_on_levels=None,
        align_qkv_bias: bool = False,
        align_offset_groups: int = 4,
        num_change_queries: int = 16,
        cqi_heads: int = 4,
        mask_dim: int = 128,
        mask_queries: int = 32,
        mask_decoder_layers: int = 3,
        mask_heads: int = 4,
        **kwargs,
    ):
        super().__init__()
        del n_layers
        self.encoder = HierarchicalCnnDinoEncoder(backbone=backbone, fpn_channels=fpn_channels, **kwargs)
        self.ha = HarmonizedAlignment(
            channels=fpn_channels,
            disable_soft_alignment=disable_soft_alignment,
            align_window=align_window,
            align_points=align_points,
            align_heads=align_heads,
            align_on_levels=align_on_levels,
            align_qkv_bias=align_qkv_bias,
            align_offset_groups=align_offset_groups,
        )
        self.cqi = ChangeQueryInteraction(
            channels=fpn_channels,
            num_queries=num_change_queries,
            num_heads=cqi_heads,
        )
        self.mask_head = Mask2FormerChangeHead(
            channels=fpn_channels,
            mask_dim=mask_dim,
            num_mask_queries=mask_queries,
            num_decoder_layers=mask_decoder_layers,
            num_heads=mask_heads,
        )
        self.aux_heads = MultiScaleAuxiliaryHead(fpn_channels)

    @torch.inference_mode()
    def predict_logits(self, x1, x2):
        logits, _ = self.forward(x1, x2)
        return logits

    def forward(self, x1, x2, gt_mask=None, current_epoch: int | None = None):
        del gt_mask, current_epoch
        pre_pyramid = self.encoder(x1)
        post_pyramid = self.encoder(x2)
        aligned_pre, aligned_post = self.ha(pre_pyramid, post_pyramid)
        change_primitives = self.cqi(aligned_pre, aligned_post)
        final_pred = self.mask_head(change_primitives, x1.shape[-2:])
        aux_preds = self.aux_heads(change_primitives, x1.shape[-2:])
        return final_pred, aux_preds
