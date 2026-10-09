import torch.nn as nn
import torch.nn.functional as F

from ..necks import DsBnRelu


class MultiScaleAuxiliaryHead(nn.Module):
    """P1-P5 辅助监督头，兼顾小目标和大范围洪水区域。"""

    def __init__(self, channels: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [nn.Sequential(DsBnRelu(channels, channels), nn.Conv2d(channels, 2, 1)) for _ in range(5)]
        )

    def forward(self, features, output_size: tuple[int, int]):
        return tuple(
            F.interpolate(head(feat), size=output_size, mode="bilinear", align_corners=False)
            for head, feat in zip(self.heads, features)
        )
