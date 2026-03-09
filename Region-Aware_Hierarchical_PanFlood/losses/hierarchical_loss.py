from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def binary_dice_loss(logits: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None, eps: float = 1e-6):
    probs = torch.sigmoid(logits)
    target = target.float()
    if mask is None:
        mask = torch.ones_like(target, dtype=torch.bool)

    probs = probs[mask]
    target = target[mask]
    if probs.numel() == 0:
        return logits.new_tensor(0.0)

    inter = (probs * target).sum()
    union = probs.sum() + target.sum()
    return 1.0 - (2.0 * inter + eps) / (union + eps)


def multiclass_dice_loss(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = 255, eps: float = 1e-6):
    n_cls = logits.shape[1]
    probs = torch.softmax(logits, dim=1)
    valid = target != ignore_index
    if valid.sum() == 0:
        return logits.new_tensor(0.0)

    loss = logits.new_tensor(0.0)
    cnt = 0
    for c in range(n_cls):
        t = (target == c).float()
        p = probs[:, c]
        p = p[valid]
        t = t[valid]
        inter = (p * t).sum()
        union = p.sum() + t.sum()
        loss = loss + (1.0 - (2.0 * inter + eps) / (union + eps))
        cnt += 1
    return loss / max(cnt, 1)


def _edge_map(binary_mask: torch.Tensor) -> torch.Tensor:
    # 轻量边界近似：maxpool 与原图差值。
    pooled = F.max_pool2d(binary_mask, kernel_size=3, stride=1, padding=1)
    edge = (pooled - binary_mask).clamp(min=0.0, max=1.0)
    return edge


class RegionAwareHierLoss(nn.Module):
    """分层损失：floodness + router + experts + final + refinement + consistency。"""

    def __init__(self, cfg: Dict):
        super().__init__()
        loss_cfg = cfg["loss"]
        data_cfg = cfg["data"]

        self.ignore_index = int(data_cfg.get("ignore_index", 255))
        self.urban_class_weight = float(loss_cfg.get("urban_class_weight", 4.0))

        self.w_floodness = float(loss_cfg.get("w_floodness", 1.0))
        self.w_router = float(loss_cfg.get("w_router", 0.3))
        self.w_open = float(loss_cfg.get("w_open", 0.5))
        self.w_urban = float(loss_cfg.get("w_urban", 0.8))
        self.w_final = float(loss_cfg.get("w_final", 1.0))
        self.w_boundary = float(loss_cfg.get("w_boundary", 0.2))
        self.w_consistency = float(loss_cfg.get("w_consistency", 0.1))

        self.use_focal_final = bool(loss_cfg.get("use_focal_final", True))
        self.focal_gamma = float(loss_cfg.get("focal_gamma", 2.0))
        self.focal_alpha = float(loss_cfg.get("focal_alpha", 0.25))

    def _router_target(self, y: torch.Tensor) -> torch.Tensor:
        # 0->ambiguous, 1->open-like, 2->urban-like
        t = torch.full_like(y, 2)
        t[y == 1] = 0
        t[y == 2] = 1
        t[y == self.ignore_index] = self.ignore_index
        return t

    def _focal_ce(self, logits: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None):
        ce = F.cross_entropy(logits, target, weight=weight, ignore_index=self.ignore_index, reduction="none")
        pt = torch.exp(-ce)
        loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce
        valid = target != self.ignore_index
        if valid.sum() == 0:
            return logits.new_tensor(0.0)
        return loss[valid].mean()

    def forward(self, out: Dict[str, torch.Tensor], target: torch.Tensor) -> Dict[str, torch.Tensor]:
        valid = target != self.ignore_index
        flood_target = (target > 0).float()

        # 1) floodness
        floodness_logits = out["floodness_logits"].squeeze(1)
        bce = F.binary_cross_entropy_with_logits(floodness_logits[valid], flood_target[valid]) if valid.sum() > 0 else floodness_logits.new_tensor(0.0)
        dice_f = binary_dice_loss(floodness_logits, flood_target, mask=valid)
        loss_floodness = bce + dice_f

        # 2) router
        router_target = self._router_target(target)
        loss_router = F.cross_entropy(out["router_logits"], router_target, ignore_index=self.ignore_index)

        # 3) open expert (class-1 正样本，class-2忽略)
        open_logits = out["open_logits"].squeeze(1)
        open_mask = (target != 2) & valid
        open_target = (target == 1).float()
        if open_mask.sum() > 0:
            loss_open_bce = F.binary_cross_entropy_with_logits(open_logits[open_mask], open_target[open_mask])
        else:
            loss_open_bce = open_logits.new_tensor(0.0)
        loss_open_dice = binary_dice_loss(open_logits, open_target, mask=open_mask)
        loss_open = loss_open_bce + loss_open_dice

        # 4) urban expert (class-2 正样本，class-1忽略)
        urban_logits = out["urban_logits"].squeeze(1)
        urban_mask = (target != 1) & valid
        urban_target = (target == 2).float()
        if urban_mask.sum() > 0:
            pos_weight = urban_logits.new_tensor(self.urban_class_weight)
            loss_urban_bce = F.binary_cross_entropy_with_logits(
                urban_logits[urban_mask], urban_target[urban_mask], pos_weight=pos_weight
            )
        else:
            loss_urban_bce = urban_logits.new_tensor(0.0)
        loss_urban_dice = binary_dice_loss(urban_logits, urban_target, mask=urban_mask)
        loss_urban = loss_urban_bce + loss_urban_dice

        # 5) final segmentation
        class_weights = out["final_logits"].new_tensor([1.0, 1.2, self.urban_class_weight])
        if self.use_focal_final:
            loss_final_ce = self._focal_ce(out["final_logits"], target, weight=class_weights)
        else:
            loss_final_ce = F.cross_entropy(out["final_logits"], target, weight=class_weights, ignore_index=self.ignore_index)
        loss_final_dice = multiclass_dice_loss(out["final_logits"], target, ignore_index=self.ignore_index)
        loss_final = loss_final_ce + loss_final_dice

        # 6) boundary/refinement
        flood_pred = torch.softmax(out["final_logits"], dim=1)[:, 1:3].sum(dim=1, keepdim=True)
        edge_pred = _edge_map(flood_pred)
        edge_gt = _edge_map(flood_target.unsqueeze(1))
        loss_boundary = F.l1_loss(edge_pred[valid.unsqueeze(1)], edge_gt[valid.unsqueeze(1)]) if valid.sum() > 0 else flood_pred.new_tensor(0.0)

        # 7) optional consistency
        state_maps = out.get("state_maps", {})
        if state_maps:
            e_open = state_maps["delta_intensity"].abs()
            e_urban = state_maps["delta_coherence"].abs()
            e_open = e_open / (e_open.amax(dim=(-2, -1), keepdim=True) + 1e-6)
            e_urban = e_urban / (e_urban.amax(dim=(-2, -1), keepdim=True) + 1e-6)
            loss_cons_open = F.mse_loss(torch.sigmoid(open_logits).unsqueeze(1), e_open)
            loss_cons_urban = F.mse_loss(torch.sigmoid(urban_logits).unsqueeze(1), e_urban)
            loss_consistency = 0.5 * (loss_cons_open + loss_cons_urban)
        else:
            loss_consistency = out["final_logits"].new_tensor(0.0)

        total = (
            self.w_floodness * loss_floodness
            + self.w_router * loss_router
            + self.w_open * loss_open
            + self.w_urban * loss_urban
            + self.w_final * loss_final
            + self.w_boundary * loss_boundary
            + self.w_consistency * loss_consistency
        )

        return {
            "loss_total": total,
            "loss_floodness": loss_floodness,
            "loss_router": loss_router,
            "loss_open": loss_open,
            "loss_urban": loss_urban,
            "loss_final": loss_final,
            "loss_boundary": loss_boundary,
            "loss_consistency": loss_consistency,
        }
