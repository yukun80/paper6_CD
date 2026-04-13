"""轻量末端精修器。

默认使用像素级 contrast refiner 修复 tiny flood；若用户显式选择，
也支持 topology-only 或 hybrid 模式。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .topo_router import FloodTopoRouter


class ContrastRefiner(nn.Module):
    """像素级 tiny-aware 精修器。

    只消费 Detector 的高分辨率细节特征和 tiny prior，不再做图级聚合，
    避免微小洪水目标在网格节点中被提前稀释。
    """

    def __init__(self, feat_dim: int, proj_dim: int = 32):
        super().__init__()
        self.feat_proj = nn.Sequential(
            nn.Conv2d(feat_dim, proj_dim, 1, bias=False),
            nn.BatchNorm2d(proj_dim),
            nn.SiLU(inplace=True),
        )
        self.refine_head = nn.Sequential(
            nn.Conv2d(proj_dim + 2, proj_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(proj_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(proj_dim, 1, 1, bias=True),
        )
        self.alpha_raw = nn.Parameter(torch.tensor(-3.0))

    def forward(
        self,
        logit_2ch: torch.Tensor,
        det_feat: torch.Tensor,
        tiny_prior_map: torch.Tensor | None = None,
        gt_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """利用像素级细节先验做双向校正。

        `gt_mask` 仅为统一接口保留，不参与 contrast refiner 计算。
        """
        del gt_mask
        _, _, H, W = logit_2ch.shape
        p_fg = F.softmax(logit_2ch, dim=1)[:, 1:2]

        feat_up = F.interpolate(
            self.feat_proj(det_feat), size=(H, W), mode="bilinear", align_corners=False
        )
        if tiny_prior_map is None:
            tiny_prior_map = torch.zeros_like(p_fg)
        else:
            tiny_prior_map = F.interpolate(
                tiny_prior_map, size=(H, W), mode="bilinear", align_corners=False
            )

        correction = torch.tanh(
            self.refine_head(torch.cat([feat_up, p_fg, tiny_prior_map], dim=1))
        )
        alpha = torch.sigmoid(self.alpha_raw)
        corrected_p = (p_fg + alpha * correction).clamp(1e-6, 1 - 1e-6)
        corrected_logit = torch.logit(corrected_p, eps=1e-6)

        out = logit_2ch.clone()
        out[:, 1:2] = out[:, 1:2] + alpha * (corrected_logit - out[:, 1:2])
        return out, None


class HybridRefiner(nn.Module):
    """先做像素级精修，再在 contrast/topo 一致的大区域上施加拓扑约束。"""

    def __init__(self, feat_dim: int, topo_kwargs: dict[str, int | float]):
        super().__init__()
        self.contrast_refiner = ContrastRefiner(feat_dim=feat_dim)
        self.topo_refiner = FloodTopoRouter(feat_dim=feat_dim, **topo_kwargs)
        self.large_region_thresh = 0.35
        self.large_region_kernel = 17

    def forward(
        self,
        logit_p1_2ch: torch.Tensor,
        det_p1_feat: torch.Tensor,
        logit_p2_2ch: torch.Tensor,
        det_p2_feat: torch.Tensor,
        tiny_prior_map: torch.Tensor | None = None,
        risk_map: torch.Tensor | None = None,
        gt_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        contrast_out, _ = self.contrast_refiner(
            logit_p1_2ch, det_p1_feat, tiny_prior_map=tiny_prior_map
        )
        topo_out, topo_loss = self.topo_refiner(
            logit_p2_2ch, det_p2_feat, risk_map=risk_map, gt_mask=gt_mask
        )

        contrast_fg = F.softmax(contrast_out, dim=1)[:, 1:2]
        topo_fg = F.softmax(topo_out, dim=1)[:, 1:2]
        pad = self.large_region_kernel // 2
        contrast_density = F.avg_pool2d(
            contrast_fg, self.large_region_kernel, stride=1, padding=pad
        )
        topo_density = F.avg_pool2d(
            topo_fg, self.large_region_kernel, stride=1, padding=pad
        )
        agreement = 1.0 - torch.abs(contrast_fg - topo_fg)
        agreement_density = F.avg_pool2d(
            agreement, self.large_region_kernel, stride=1, padding=pad
        ).clamp(0.0, 1.0)
        contrast_gate = torch.sigmoid((contrast_density - self.large_region_thresh) * 12.0)
        topo_gate = torch.sigmoid((topo_density - self.large_region_thresh) * 12.0)
        agreement_gate = torch.sigmoid((agreement_density - 0.70) * 12.0)
        large_gate = contrast_gate * topo_gate * agreement_gate
        if tiny_prior_map is not None:
            tiny_gate = F.interpolate(
                tiny_prior_map, size=contrast_fg.shape[-2:], mode="bilinear", align_corners=False
            ).clamp(0.0, 1.0)
            large_gate = large_gate * (1.0 - tiny_gate)
        blended_fg = torch.lerp(contrast_fg, topo_fg, large_gate).clamp(1e-6, 1 - 1e-6)

        out = contrast_out.clone()
        out[:, 1:2] = torch.logit(blended_fg, eps=1e-6)
        return out, topo_loss
