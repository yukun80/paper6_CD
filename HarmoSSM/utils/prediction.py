"""主工程验证及推理共用的 FP64 概率与阈值比较契约。"""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np

if TYPE_CHECKING:
    import torch


def foreground_probability(logits: torch.Tensor) -> torch.Tensor:
    """仅将输出 logits 转 FP64 再 softmax，不改变网络或损失精度。"""
    return logits.double().softmax(dim=1)[:, 1]


@overload
def threshold_probability(probabilities: np.ndarray, threshold: float) -> np.ndarray: ...


@overload
def threshold_probability(probabilities: torch.Tensor, threshold: float) -> torch.Tensor: ...


def threshold_probability(
    probabilities: np.ndarray | torch.Tensor, threshold: float,
) -> np.ndarray | torch.Tensor:
    """概率和比较阈值均显式使用 float64；不改变形状、设备或有效区。"""
    comparison_threshold = np.float64(threshold)
    if isinstance(probabilities, np.ndarray):
        return probabilities.astype(np.float64, copy=False) >= comparison_threshold
    probability = probabilities.double()
    return probability >= probability.new_tensor(float(comparison_threshold))
