import torch
import torch.nn as nn
import torch.nn.functional as F

from .deformable_alignment import DeformableAlignmentBlock


class PairSharedStyleCalibration(nn.Module):
    """PSC：用 pair-shared 统计和源域原型协调 SAR 浅层风格。"""

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.source_mu = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.source_log_std = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.recover = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        nn.init.zeros_(self.recover[-2].weight)

    def _calibrate_one(self, feat: torch.Tensor, pair_mu: torch.Tensor, pair_std: torch.Tensor) -> torch.Tensor:
        source_std = F.softplus(self.source_log_std) + self.eps
        normalized = (feat - pair_mu) / pair_std
        calibrated = normalized * source_std + self.source_mu
        return calibrated + self.recover(calibrated)

    def forward(self, pre_feat: torch.Tensor, post_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pair = torch.stack([pre_feat, post_feat], dim=1)
        pair_mu = pair.mean(dim=(1, 3, 4), keepdim=False).unsqueeze(-1).unsqueeze(-1)
        pair_var = pair.var(dim=(1, 3, 4), keepdim=False, unbiased=False).unsqueeze(-1).unsqueeze(-1)
        pair_std = torch.sqrt(pair_var + self.eps)
        return (
            self._calibrate_one(pre_feat, pair_mu, pair_std),
            self._calibrate_one(post_feat, pair_mu, pair_std),
        )


class HarmonizedAlignment(nn.Module):
    """Module II：HA 输出可比较双时相特征，而不直接做变化判别。"""

    def __init__(
        self,
        channels: int,
        disable_soft_alignment: bool = False,
        align_window: int = 5,
        align_points: int = 9,
        align_heads: int = 4,
        align_on_levels: list[int] | tuple[int, ...] | None = None,
        align_qkv_bias: bool = False,
        align_offset_groups: int = 4,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        if align_on_levels is None:
            align_on_levels = [1, 2, 3]
        if disable_soft_alignment:
            align_on_levels = []
        self.align_on_levels = {int(level) for level in align_on_levels}
        invalid = sorted(level for level in self.align_on_levels if level not in {1, 2, 3})
        if invalid:
            raise ValueError(f"HA alignment only supports P1/P2/P3, got {invalid}")

        self.psc_p1 = PairSharedStyleCalibration(channels)
        self.psc_p2 = PairSharedStyleCalibration(channels)
        self.align_p1 = self._make_alignment(
            channels,
            align_heads=max(1, align_heads // 2),
            align_points=max(4, align_points // 2),
            align_window=align_window,
            align_offset_groups=max(1, align_offset_groups // 2),
            align_qkv_bias=align_qkv_bias,
            enabled=1 in self.align_on_levels,
        )
        self.align_p2 = self._make_alignment(
            channels,
            align_heads=align_heads,
            align_points=align_points,
            align_window=align_window,
            align_offset_groups=align_offset_groups,
            align_qkv_bias=align_qkv_bias,
            enabled=2 in self.align_on_levels,
        )
        self.align_p3 = self._make_alignment(
            channels,
            align_heads=align_heads,
            align_points=align_points,
            align_window=align_window,
            align_offset_groups=align_offset_groups,
            align_qkv_bias=align_qkv_bias,
            enabled=3 in self.align_on_levels,
        )

    @staticmethod
    def _make_alignment(
        channels: int,
        align_heads: int,
        align_points: int,
        align_window: int,
        align_offset_groups: int,
        align_qkv_bias: bool,
        enabled: bool,
    ) -> nn.Module | None:
        if not enabled:
            return None
        return DeformableAlignmentBlock(
            dim=channels,
            num_heads=align_heads,
            num_points=align_points,
            window_size=align_window,
            offset_groups=align_offset_groups,
            qkv_bias=align_qkv_bias,
        )

    @staticmethod
    def _align_pre_feature(
        pre_feat: torch.Tensor, post_feat: torch.Tensor, aligner: nn.Module | None
    ) -> torch.Tensor:
        return aligner(pre_feat, post_feat) if aligner is not None else pre_feat

    def forward(self, pre_pyramid, post_pyramid):
        pre_p1, pre_p2, pre_p3, pre_p4, pre_p5 = pre_pyramid
        post_p1, post_p2, post_p3, post_p4, post_p5 = post_pyramid

        pre_p1, post_p1 = self.psc_p1(pre_p1, post_p1)
        pre_p2, post_p2 = self.psc_p2(pre_p2, post_p2)

        aligned_pre = (
            self._align_pre_feature(pre_p1, post_p1, self.align_p1),
            self._align_pre_feature(pre_p2, post_p2, self.align_p2),
            self._align_pre_feature(pre_p3, post_p3, self.align_p3),
            pre_p4,
            pre_p5,
        )
        aligned_post = (post_p1, post_p2, post_p3, post_p4, post_p5)
        return aligned_pre, aligned_post
