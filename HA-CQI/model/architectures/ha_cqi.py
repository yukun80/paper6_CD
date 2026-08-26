import torch
import torch.nn as nn

from ..decode_heads import MultiScaleAuxiliaryHead, OmniScaleStateSpaceChangeDecoder
from ..modules import ChangeQueryInteraction, HarmonizedAlignment, HierarchicalCnnDinoEncoder


class HACQIModel(nn.Module):
    """HA-CQI 主模型：Encoder -> HA -> CQI -> omni-scale dense decoder。"""

    def __init__(
        self,
        fpn_channels: int = 128,
        disable_soft_alignment: bool = False,
        align_window: int = 5,
        align_points: int = 9,
        align_heads: int = 4,
        align_on_levels=None,
        align_qkv_bias: bool = False,
        align_offset_groups: int = 4,
        num_change_queries: int = 16,
        cqi_heads: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.encoder = HierarchicalCnnDinoEncoder(fpn_channels=fpn_channels, **kwargs)
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
        self.decoder = OmniScaleStateSpaceChangeDecoder(
            channels=fpn_channels,
            d_state=1,
            ffn_ratio=4,
        )
        self.aux_heads = MultiScaleAuxiliaryHead(fpn_channels)

    def extract_change_features(self, x1, x2) -> dict[str, tuple[torch.Tensor, ...]]:
        """导出可诊断的多尺度特征；训练主路径与诊断共用同一计算。"""
        pre_pyramid = self.encoder(x1)
        post_pyramid = self.encoder(x2)
        aligned_pre, aligned_post = self.ha(pre_pyramid, post_pyramid)
        change_primitives = self.cqi(aligned_pre, aligned_post)
        return {
            "pre_pyramid": pre_pyramid,
            "post_pyramid": post_pyramid,
            "aligned_pre": aligned_pre,
            "aligned_post": aligned_post,
            "change_primitives": change_primitives,
        }

    @torch.inference_mode()
    def predict_logits(self, x1, x2):
        logits, _ = self.forward(x1, x2)
        return logits

    def forward(self, x1, x2):
        feature_bundle = self.extract_change_features(x1, x2)
        change_primitives = feature_bundle["change_primitives"]
        final_pred = self.decoder(change_primitives, x1.shape[-2:])
        aux_preds = self.aux_heads(change_primitives, x1.shape[-2:])
        return final_pred, aux_preds
