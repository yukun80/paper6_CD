"""二值分割指标。"""

from __future__ import annotations

import torch


def _prepare_binary_tensors(pred: torch.Tensor, target: torch.Tensor, ignore_index: int = 255):
    pred = pred.long()
    target = target.long()
    valid = target != int(ignore_index)
    if valid.sum() == 0:
        zero = pred.new_tensor(0.0, dtype=torch.float32)
        return zero, zero, zero
    pred = pred[valid] > 0
    target = target[valid] > 0
    intersection = (pred & target).sum().float()
    pred_sum = pred.sum().float()
    target_sum = target.sum().float()
    return intersection, pred_sum, target_sum


def binary_iou(pred: torch.Tensor, target: torch.Tensor, ignore_index: int = 255) -> float:
    """计算二值 IoU。"""
    inter, pred_sum, target_sum = _prepare_binary_tensors(pred, target, ignore_index)
    union = pred_sum + target_sum - inter
    if union.item() <= 0:
        return 1.0
    return float((inter / union).item())


def binary_dice(pred: torch.Tensor, target: torch.Tensor, ignore_index: int = 255) -> float:
    """计算二值 Dice/F1。"""
    inter, pred_sum, target_sum = _prepare_binary_tensors(pred, target, ignore_index)
    denom = pred_sum + target_sum
    if denom.item() <= 0:
        return 1.0
    return float((2.0 * inter / denom).item())
