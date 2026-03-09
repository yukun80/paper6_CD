"""可视化相关工具。"""

from __future__ import annotations

import numpy as np


DEFAULT_PALETTE = {
    0: (0, 0, 0),
    1: (255, 64, 64),
    2: (64, 220, 255),
}
IGNORE_COLOR = (160, 160, 160)


def colorize_label(label: np.ndarray, ignore_index: int = 255) -> np.ndarray:
    """将整型标签转为伪彩色图。"""

    h, w = label.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in DEFAULT_PALETTE.items():
        out[label == cls] = np.array(color, dtype=np.uint8)
    out[label == ignore_index] = np.array(IGNORE_COLOR, dtype=np.uint8)
    return out


def to_uint8_gray(arr: np.ndarray, low_q: float = 2.0, high_q: float = 98.0) -> np.ndarray:
    """将浮点数组按分位数拉伸到 0-255。"""

    arr = arr.astype(np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr, dtype=np.uint8)

    v = arr[finite]
    lo = np.percentile(v, low_q)
    hi = np.percentile(v, high_q)
    if hi <= lo:
        hi = lo + 1e-6

    norm = (arr - lo) / (hi - lo)
    norm = np.clip(norm, 0.0, 1.0)
    norm[~finite] = 0.0
    return (norm * 255.0).astype(np.uint8)


def make_overlay(gray: np.ndarray, color: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """将灰度底图与伪彩图叠加。"""

    if gray.ndim == 2:
        gray3 = np.stack([gray, gray, gray], axis=-1)
    else:
        gray3 = gray
    out = gray3.astype(np.float32) * (1.0 - alpha) + color.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)
