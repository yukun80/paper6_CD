"""预测输出共用的 FP32 概率与阈值比较契约。"""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np

if TYPE_CHECKING:
    import torch


def foreground_probability(logits: torch.Tensor) -> torch.Tensor:
    """先转 FP32 再 softmax，返回同设备的 B×H×W 前景概率。"""
    return logits.float().softmax(dim=1)[:, 1]


@overload
def threshold_probability(probabilities: np.ndarray, threshold: float) -> np.ndarray: ...


@overload
def threshold_probability(probabilities: torch.Tensor, threshold: float) -> torch.Tensor: ...


def threshold_probability(
    probabilities: np.ndarray | torch.Tensor, threshold: float,
) -> np.ndarray | torch.Tensor:
    """概率和比较阈值均显式使用 float32；不改变形状、设备或有效区。"""
    comparison_threshold = np.float32(threshold)
    if isinstance(probabilities, np.ndarray):
        return probabilities.astype(np.float32, copy=False) >= comparison_threshold
    probability = probabilities.float()
    return probability >= probability.new_tensor(float(comparison_threshold))
