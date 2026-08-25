"""面向双时相 SAR 变化原语的 Omni-Scale State-Space Change Decoder。"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from ..necks import DsBnRelu
from .state_space_scan import StateSpaceContextBlock


class ConvNormAct(nn.Sequential):
    """轻量卷积投影，统一 decoder 内部的通道契约。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class OmniScaleStateSpaceChangeDecoder(nn.Module):
    """P3-P5 建模全局上下文，P2/P1 仅用于逐级细节重建。"""

    def __init__(self, channels: int = 128, d_state: int = 1, ffn_ratio: int = 4):
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        self.channels = int(channels)
        self.d_state = int(d_state)
        scan_channels = 3 * self.channels

        self.context_projections = nn.ModuleList(
            [ConvNormAct(self.channels, self.channels) for _ in range(3)]
        )
        self.context_fusion = ConvNormAct(3 * self.channels, self.channels)

        # MMSCoPE 的多尺度区域聚合：保持三条路径的有效感受野差异。
        self.region_stride2 = ConvNormAct(
            self.channels,
            2 * self.channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.region_stride4 = ConvNormAct(
            self.channels,
            4 * self.channels,
            kernel_size=5,
            stride=4,
            padding=1,
        )
        self.region_reductions = nn.ModuleList(
            [
                ConvNormAct(16 * self.channels, self.channels),
                ConvNormAct(8 * self.channels, self.channels),
                ConvNormAct(4 * self.channels, self.channels),
            ]
        )
        self.omni_context = StateSpaceContextBlock(
            scan_channels,
            d_state=self.d_state,
            ffn_ratio=ffn_ratio,
        )
        self.scan_reinjection = ConvNormAct(scan_channels, self.channels)

        self.short_path = ConvNormAct(self.channels, self.channels)
        self.global_path = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.channels, self.channels, 1, bias=True),
            nn.SiLU(inplace=True),
        )
        # raw scan(3C) + short/global(2C) + P3/P4/P5 context(3C) = 8C。
        self.context_reconstruction = ConvNormAct(8 * self.channels, self.channels)

        self.detail_projections = nn.ModuleList(
            [ConvNormAct(self.channels, self.channels) for _ in range(2)]
        )
        self.p2_reconstruction = nn.Sequential(
            DsBnRelu(2 * self.channels, self.channels),
            DsBnRelu(self.channels, self.channels),
        )
        self.p1_reconstruction = nn.Sequential(
            DsBnRelu(2 * self.channels, self.channels),
            DsBnRelu(self.channels, self.channels),
        )
        self.flood_head = nn.Sequential(
            DsBnRelu(self.channels, self.channels),
            nn.Conv2d(self.channels, 2, 1),
        )

        self._last_diagnostics: dict[str, torch.Tensor] = {}
        self._last_shapes: dict[str, tuple[int, ...]] = {}

    @staticmethod
    def _resize(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        if x.shape[-2:] == size:
            return x
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    @staticmethod
    def _rms(x: torch.Tensor) -> torch.Tensor:
        return x.detach().float().square().mean().sqrt()

    @staticmethod
    def _pad_to_multiple_of_four(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        height, width = x.shape[-2:]
        pad_height = (-height) % 4
        pad_width = (-width) % 4
        if pad_height or pad_width:
            x = F.pad(x, (0, pad_width, 0, pad_height), mode="replicate")
        return x, pad_height, pad_width

    def diagnostics(self) -> dict[str, torch.Tensor]:
        """返回最近一次 forward 的幅值诊断，不持有计算图。"""
        if self._last_diagnostics:
            return dict(self._last_diagnostics)
        zero = next(self.parameters()).new_zeros(())
        return {
            "decoder_scan_rms": zero,
            "decoder_context_rms": zero,
            "decoder_p2_detail_rms": zero,
            "decoder_p1_detail_rms": zero,
            "decoder_context_detail_ratio": zero,
        }

    def last_feature_shapes(self) -> dict[str, tuple[int, ...]]:
        """仅供结构验收读取最近一次中间 tensor shape。"""
        return dict(self._last_shapes)

    def _build_context(self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor):
        p3_size = p3.shape[-2:]
        stages = [
            projection(feature)
            for projection, feature in zip(self.context_projections, (p3, p4, p5))
        ]
        stages = [self._resize(feature, p3_size) for feature in stages]
        fused = self.context_fusion(torch.cat(stages, dim=1))
        fused_pad, pad_height, pad_width = self._pad_to_multiple_of_four(fused)

        region_full = fused_pad
        region_half = self.region_stride2(fused_pad)
        region_quarter = self.region_stride4(fused_pad)
        scan_inputs = (
            self.region_reductions[0](F.pixel_unshuffle(region_full, 4)),
            self.region_reductions[1](F.pixel_unshuffle(region_half, 2)),
            self.region_reductions[2](region_quarter),
        )
        scan_input = torch.cat(scan_inputs, dim=1)
        scan_output = self.omni_context(scan_input)
        scan_up = self._resize(scan_output, fused_pad.shape[-2:])
        reinjected = self.scan_reinjection(scan_up)

        padded_size = fused_pad.shape[-2:]
        padded_stages = [
            F.pad(stage, (0, pad_width, 0, pad_height), mode="replicate")
            if pad_height or pad_width
            else stage
            for stage in stages
        ]
        if any(stage.shape[-2:] != padded_size for stage in padded_stages):
            raise RuntimeError("OSCD context padding produced inconsistent stage shapes")
        global_context = self._resize(self.global_path(fused_pad), padded_size)
        reconstruction_input = torch.cat(
            [
                scan_up,
                self.short_path(fused_pad),
                global_context,
                *(stage + reinjected for stage in padded_stages),
            ],
            dim=1,
        )
        context = self.context_reconstruction(reconstruction_input)
        if pad_height:
            context = context[..., :-pad_height, :]
        if pad_width:
            context = context[..., :, :-pad_width]

        self._last_shapes = {
            "fused_p3": tuple(fused.shape),
            "region_full": tuple(region_full.shape),
            "region_half": tuple(region_half.shape),
            "region_quarter": tuple(region_quarter.shape),
            "scan_input": tuple(scan_input.shape),
            "scan_output": tuple(scan_output.shape),
        }
        return context, scan_output

    def forward(self, features, output_size: tuple[int, int]) -> torch.Tensor:
        if len(features) != 5:
            raise ValueError(f"OSCD expects P1-P5, got {len(features)} feature levels")
        if any(feature.ndim != 4 for feature in features):
            raise ValueError("Every OSCD feature must be a BCHW tensor")
        if any(feature.shape[1] != self.channels for feature in features):
            shapes = [tuple(feature.shape) for feature in features]
            raise ValueError(f"OSCD expects {self.channels} channels at every level, got {shapes}")

        p1, p2, p3, p4, p5 = features
        r3, scan_output = self._build_context(p3, p4, p5)
        p2_detail = self.detail_projections[0](p2)
        r2 = self.p2_reconstruction(
            torch.cat((self._resize(r3, p2.shape[-2:]), p2_detail), dim=1)
        )
        p1_detail = self.detail_projections[1](p1)
        r1 = self.p1_reconstruction(
            torch.cat((self._resize(r2, p1.shape[-2:]), p1_detail), dim=1)
        )
        logits = self.flood_head(r1)

        scan_rms = self._rms(scan_output)
        context_rms = self._rms(r3)
        p2_rms = self._rms(p2_detail)
        p1_rms = self._rms(p1_detail)
        detail_rms = 0.5 * (p2_rms + p1_rms)
        self._last_diagnostics = {
            "decoder_scan_rms": scan_rms,
            "decoder_context_rms": context_rms,
            "decoder_p2_detail_rms": p2_rms,
            "decoder_p1_detail_rms": p1_rms,
            "decoder_context_detail_ratio": context_rms / detail_rms.clamp_min(1e-8),
        }
        return self._resize(logits, output_size)
