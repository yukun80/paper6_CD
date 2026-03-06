from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class PPM(nn.Module):
    """金字塔池化模块，用于增强最高层特征的全局上下文。"""

    def __init__(self, in_channels: int, channels: int, pool_scales: Sequence[int] = (1, 2, 3, 6)) -> None:
        super().__init__()
        self.pool_scales = list(pool_scales)
        self.paths = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    nn.Conv2d(in_channels, channels, kernel_size=1, bias=False),
                    nn.ReLU(inplace=True),
                )
                for scale in self.pool_scales
            ]
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + len(self.pool_scales) * channels, channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = [x]
        size = x.shape[-2:]
        for path in self.paths:
            y = path(x)
            y = F.interpolate(y, size=size, mode="bilinear", align_corners=False)
            out.append(y)
        return self.bottleneck(torch.cat(out, dim=1))


class SharedUPerNeck(nn.Module):
    """轻量 UPerNet 风格共享 neck：PPM + FPN，输出共享特征图 G。"""

    def __init__(
        self,
        in_channels: int,
        num_levels: int,
        fpn_dim: int = 256,
        ppm_scales: Sequence[int] = (1, 2, 3, 6),
    ) -> None:
        super().__init__()
        self.num_levels = int(num_levels)

        self.lateral_convs = nn.ModuleList([nn.Conv2d(in_channels, fpn_dim, kernel_size=1) for _ in range(self.num_levels)])
        self.fpn_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                )
                for _ in range(self.num_levels)
            ]
        )
        self.ppm = PPM(fpn_dim, fpn_dim, pool_scales=ppm_scales)
        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(self.num_levels * fpn_dim, fpn_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        if len(feats) != self.num_levels:
            raise ValueError(f"Expected {self.num_levels} features, got {len(feats)}")

        # 1) 各层统一到 fpn_dim。
        laterals = [conv(x) for conv, x in zip(self.lateral_convs, feats)]

        # 2) 最高层做 PPM。
        laterals[-1] = self.ppm(laterals[-1])

        # 3) FPN 自顶向下融合。
        for i in range(self.num_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[-2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=prev_shape, mode="bilinear", align_corners=False
            )

        # 4) 各层平滑后上采样到最高分辨率并拼接融合。
        fpn_outs = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]
        base_size = fpn_outs[0].shape[-2:]
        fpn_outs = [F.interpolate(x, size=base_size, mode="bilinear", align_corners=False) for x in fpn_outs]
        return self.fpn_bottleneck(torch.cat(fpn_outs, dim=1))


class BinarySegHead(nn.Module):
    """稳定的二分类分割头：Conv-BN-ReLU x2 + 1x1 conv。"""

    def __init__(self, in_channels: int, hidden_channels: int = 128) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.block(x))
