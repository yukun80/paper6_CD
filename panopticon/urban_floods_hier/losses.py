from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_mean(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x.new_tensor(0.0)
    return x.mean()


def _binary_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """二分类 Dice，按 mask 过滤有效像素。"""
    probs = torch.sigmoid(logits)
    probs = probs.squeeze(1)

    valid = mask.bool()
    if valid.sum() == 0:
        return logits.new_tensor(0.0)

    p = probs[valid]
    t = target.float()[valid]
    inter = (p * t).sum()
    denom = p.sum() + t.sum()
    return 1.0 - (2.0 * inter + eps) / (denom + eps)


def _binary_bce_or_focal(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    kind: str,
    focal_alpha: float,
    focal_gamma: float,
) -> torch.Tensor:
    """支持 BCE 与 Focal BCE 的二分类损失。"""
    logits = logits.squeeze(1)
    valid = mask.bool()
    if valid.sum() == 0:
        return logits.new_tensor(0.0)

    l = logits[valid]
    t = target.float()[valid]

    bce = F.binary_cross_entropy_with_logits(l, t, reduction="none")
    if kind == "bce":
        return _safe_mean(bce)
    if kind != "focal_bce":
        raise ValueError(f"Unsupported type loss kind: {kind}")

    p = torch.sigmoid(l)
    pt = torch.where(t > 0.5, p, 1.0 - p)
    mod = (1.0 - pt).pow(focal_gamma)
    if focal_alpha >= 0:
        alpha_t = torch.where(t > 0.5, torch.full_like(t, focal_alpha), torch.full_like(t, 1.0 - focal_alpha))
        mod = mod * alpha_t
    return _safe_mean(mod * bce)


class HierarchicalPanFloodLoss(nn.Module):
    """Hierarchical PanFlood-Adapter 配套损失。

    L = L_floodness + w_type * L_type + w_main * L_main
    """

    def __init__(
        self,
        ignore_index_main: int = 255,
        ignore_index_floodness: int = 255,
        ignore_index_flood_type: int = 255,
        floodness_bce_weight: float = 1.0,
        floodness_dice_weight: float = 1.0,
        type_loss_kind: str = "bce",
        type_focal_alpha: float = 0.25,
        type_focal_gamma: float = 2.0,
        main_ce_weight: float = 0.5,
        type_loss_weight: float = 0.8,
        eps: float = 1e-6,
        main_ce_class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.ignore_index_main = int(ignore_index_main)
        self.ignore_index_floodness = int(ignore_index_floodness)
        self.ignore_index_flood_type = int(ignore_index_flood_type)

        self.floodness_bce_weight = float(floodness_bce_weight)
        self.floodness_dice_weight = float(floodness_dice_weight)
        self.type_loss_kind = str(type_loss_kind)
        self.type_focal_alpha = float(type_focal_alpha)
        self.type_focal_gamma = float(type_focal_gamma)

        self.main_ce_weight = float(main_ce_weight)
        self.type_loss_weight = float(type_loss_weight)
        self.eps = float(eps)

        if main_ce_class_weights is not None:
            self.register_buffer("main_ce_class_weights", main_ce_class_weights.float())
        else:
            self.main_ce_class_weights = None

    def _main_ce_loss(
        self,
        final_logits: torch.Tensor,
        main_label: torch.Tensor,
        valid_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """主任务 CE，支持 ignore_index 与额外 valid_mask。"""
        if final_logits.ndim != 4 or final_logits.shape[1] != 3:
            raise ValueError(f"Expected final_logits [B,3,H,W], got {tuple(final_logits.shape)}")

        target = main_label.clone()
        if valid_mask is not None:
            target[~valid_mask.bool()] = self.ignore_index_main

        keep = target != self.ignore_index_main
        if keep.sum() == 0:
            return final_logits.new_tensor(0.0)

        return F.nll_loss(
            input=final_logits,
            target=target,
            weight=self.main_ce_class_weights,
            ignore_index=self.ignore_index_main,
            reduction="mean",
        )

    def forward(
        self,
        floodness_logits: torch.Tensor,
        flood_type_logits: torch.Tensor,
        final_logits: Optional[torch.Tensor],
        final_probs: Optional[torch.Tensor],
        main_label: torch.Tensor,
        floodness_label: torch.Tensor,
        flood_type_label: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """计算层次化损失并返回日志字典。"""
        if final_logits is None:
            if final_probs is None:
                raise ValueError("Either final_logits or final_probs must be provided")
            final_probs = final_probs.clamp(min=self.eps, max=1.0)
            final_logits = torch.log(final_probs)

        # Floodness loss mask
        floodness_valid = floodness_label != self.ignore_index_floodness
        if valid_mask is not None:
            floodness_valid = floodness_valid & valid_mask.bool()
        loss_floodness_bce = _binary_bce_or_focal(
            logits=floodness_logits,
            target=floodness_label,
            mask=floodness_valid,
            kind="bce",
            focal_alpha=0.0,
            focal_gamma=0.0,
        )
        loss_floodness_dice = _binary_dice_loss(
            logits=floodness_logits,
            target=floodness_label,
            mask=floodness_valid,
            eps=self.eps,
        )
        loss_floodness = self.floodness_bce_weight * loss_floodness_bce + self.floodness_dice_weight * loss_floodness_dice

        # Flood-type loss mask（只在洪水区域监督）
        type_valid = flood_type_label != self.ignore_index_flood_type
        if valid_mask is not None:
            type_valid = type_valid & valid_mask.bool()
        loss_type = _binary_bce_or_focal(
            logits=flood_type_logits,
            target=flood_type_label,
            mask=type_valid,
            kind=self.type_loss_kind,
            focal_alpha=self.type_focal_alpha,
            focal_gamma=self.type_focal_gamma,
        )

        # 主任务一致性损失
        loss_main = self._main_ce_loss(final_logits=final_logits, main_label=main_label, valid_mask=valid_mask)

        loss_total = loss_floodness + self.type_loss_weight * loss_type + self.main_ce_weight * loss_main

        return {
            "loss_total": loss_total,
            "loss_floodness": loss_floodness,
            "loss_floodness_bce": loss_floodness_bce,
            "loss_floodness_dice": loss_floodness_dice,
            "loss_type": loss_type,
            "loss_main": loss_main,
            "num_valid_floodness": floodness_valid.sum().detach(),
            "num_valid_type": type_valid.sum().detach(),
        }
