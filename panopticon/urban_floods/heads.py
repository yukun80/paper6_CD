from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PPM(nn.Module):
    """金字塔池化模块（Pyramid Pooling Module）。"""

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
        """多尺度池化后上采样回原尺度，再拼接融合。"""
        out = [x]
        size = x.shape[-2:]
        for path in self.paths:
            y = path(x)
            y = F.interpolate(y, size=size, mode="bilinear", align_corners=False)
            out.append(y)
        return self.bottleneck(torch.cat(out, dim=1))


class UPerLikeHead(nn.Module):
    """UPerNet 风格解码头：PPM + FPN，用于将 ViT 特征解码为分割图。"""

    def __init__(
        self,
        in_channels: int,
        num_levels: int,
        channels: int,
        num_classes: int,
        pool_scales: Sequence[int] = (1, 2, 3, 6),
    ) -> None:
        super().__init__()
        self.num_levels = num_levels
        self.lateral_convs = nn.ModuleList([nn.Conv2d(in_channels, channels, kernel_size=1) for _ in range(num_levels)])
        self.fpn_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                )
                for _ in range(num_levels)
            ]
        )
        self.ppm = PPM(channels, channels, pool_scales=pool_scales)
        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(num_levels * channels, channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, feats: List[torch.Tensor], out_size: Tuple[int, int]) -> torch.Tensor:
        """输入多层特征，输出与原图同分辨率的分割 logits。"""
        if len(feats) != self.num_levels:
            raise ValueError(f"Expected {self.num_levels} features, got {len(feats)}")

        # 1) 每层先做 lateral 变换到统一通道数。
        laterals = [conv(x) for conv, x in zip(self.lateral_convs, feats)]

        # 2) 最深层先过 PPM，增强多尺度上下文。
        laterals[-1] = self.ppm(laterals[-1])

        # 3) FPN 自顶向下融合。
        for i in range(self.num_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[-2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=prev_shape, mode="bilinear", align_corners=False
            )

        # 4) 每层再经 3x3 平滑卷积，并统一到最高分辨率后拼接。
        fpn_outs = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]
        base_size = fpn_outs[0].shape[-2:]
        fpn_outs = [F.interpolate(x, size=base_size, mode="bilinear", align_corners=False) for x in fpn_outs]

        # 5) 融合并分类，最后上采样到目标输出尺度。
        fused = self.fpn_bottleneck(torch.cat(fpn_outs, dim=1))
        logits = self.classifier(fused)
        logits = F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)
        return logits
