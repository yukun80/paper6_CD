"""监督参考与通用 SAR 特征构造工具。"""

from __future__ import annotations

import numpy as np


def robust_unit(arr: np.ndarray, valid_mask: np.ndarray, low_q: float = 1.0, high_q: float = 99.0) -> np.ndarray:
    """对有效像素做稳健归一化到 [0,1]。"""
    out = np.zeros_like(arr, dtype=np.float32)
    vals = arr[valid_mask]
    if vals.size == 0:
        return out
    low = float(np.percentile(vals, low_q))
    high = float(np.percentile(vals, high_q))
    if high - low < 1e-6:
        out[valid_mask] = 0.5
        return out
    out[valid_mask] = np.clip((vals - low) / (high - low), 0.0, 1.0)
    return out


def safe_log_ratio(pre: np.ndarray, post: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """在存在负值时，用平移后的对数差近似 log-ratio。"""
    out = np.zeros_like(pre, dtype=np.float32)
    vals = np.concatenate([pre[valid_mask], post[valid_mask]]) if valid_mask.any() else np.asarray([], dtype=np.float32)
    if vals.size == 0:
        return out
    shift = 0.0
    min_val = float(vals.min())
    if min_val <= -0.99:
        shift = -min_val + 1.0
    pre_s = np.maximum(pre + shift, 1e-6)
    post_s = np.maximum(post + shift, 1e-6)
    out[valid_mask] = np.log(post_s[valid_mask]) - np.log(pre_s[valid_mask])
    return out
