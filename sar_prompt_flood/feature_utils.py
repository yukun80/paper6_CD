"""监督参考与通用 SAR 特征构造工具。"""

from __future__ import annotations

import numpy as np

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None


def percentile_bounds(values: np.ndarray, low_q: float = 1.0, high_q: float = 99.0) -> tuple[float, float]:
    """计算稳健归一化边界。"""
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    low = float(np.percentile(values, low_q))
    high = float(np.percentile(values, high_q))
    if high - low < 1e-6:
        high = low + 1e-6
    return low, high


def robust_unit_from_bounds(arr: np.ndarray, valid_mask: np.ndarray, low: float, high: float) -> np.ndarray:
    """使用给定边界对数组做稳健归一化。"""
    out = np.zeros_like(arr, dtype=np.float32)
    safe_valid = valid_mask & np.isfinite(arr)
    if high - low < 1e-6:
        out[safe_valid] = 0.5
        return out
    out[safe_valid] = np.clip((arr[safe_valid] - low) / (high - low), 0.0, 1.0)
    out[~np.isfinite(out)] = 0.0
    return out


def robust_unit(arr: np.ndarray, valid_mask: np.ndarray, low_q: float = 1.0, high_q: float = 99.0) -> np.ndarray:
    """对有效像素做稳健归一化到 [0,1]。"""
    safe_valid = valid_mask & np.isfinite(arr)
    vals = arr[safe_valid]
    low, high = percentile_bounds(vals, low_q=low_q, high_q=high_q)
    return robust_unit_from_bounds(arr, safe_valid, low, high)


def joint_robust_norm(
    pre: np.ndarray,
    post: np.ndarray,
    valid_mask: np.ndarray,
    low_q: float = 1.0,
    high_q: float = 99.0,
) -> tuple[np.ndarray, np.ndarray]:
    """基于 pre/post 联合统计做共享归一化。"""
    safe_valid = valid_mask & np.isfinite(pre) & np.isfinite(post)
    vals = np.concatenate([pre[safe_valid], post[safe_valid]]) if safe_valid.any() else np.asarray([], dtype=np.float32)
    low, high = percentile_bounds(vals, low_q=low_q, high_q=high_q)
    return (
        robust_unit_from_bounds(pre, safe_valid, low, high),
        robust_unit_from_bounds(post, safe_valid, low, high),
    )


def safe_log_ratio(pre: np.ndarray, post: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """在存在负值时，用平移后的对数差近似 log-ratio。"""
    out = np.zeros_like(pre, dtype=np.float32)
    safe_valid = valid_mask & np.isfinite(pre) & np.isfinite(post)
    vals = np.concatenate([pre[safe_valid], post[safe_valid]]) if safe_valid.any() else np.asarray([], dtype=np.float32)
    if vals.size == 0:
        return out
    shift = 0.0
    min_val = float(vals.min())
    if min_val <= -0.99:
        shift = -min_val + 1.0
    pre_s = np.maximum(pre + shift, 1e-6)
    post_s = np.maximum(post + shift, 1e-6)
    out[safe_valid] = np.log(post_s[safe_valid]) - np.log(pre_s[safe_valid])
    out[~np.isfinite(out)] = 0.0
    return out


def local_change_strength(pre: np.ndarray, post: np.ndarray, valid_mask: np.ndarray, ksize: int = 9) -> np.ndarray:
    """估计双时相局部统计变化强度。"""
    pre = pre.astype(np.float32)
    post = post.astype(np.float32)
    safe_valid = valid_mask & np.isfinite(pre) & np.isfinite(post)
    if not safe_valid.any():
        return np.zeros_like(pre, dtype=np.float32)
    pre_fill = _fill_invalid(pre, safe_valid)
    post_fill = _fill_invalid(post, safe_valid)
    if cv2 is not None:
        pre_mean = cv2.blur(pre_fill, (ksize, ksize))
        post_mean = cv2.blur(post_fill, (ksize, ksize))
        pre_sq_mean = cv2.blur(pre_fill * pre_fill, (ksize, ksize))
        post_sq_mean = cv2.blur(post_fill * post_fill, (ksize, ksize))
    else:  # pragma: no cover
        pre_mean = _box_filter(pre_fill, ksize)
        post_mean = _box_filter(post_fill, ksize)
        pre_sq_mean = _box_filter(pre_fill * pre_fill, ksize)
        post_sq_mean = _box_filter(post_fill * post_fill, ksize)
    pre_std = np.sqrt(np.clip(pre_sq_mean - pre_mean * pre_mean, 0.0, None))
    post_std = np.sqrt(np.clip(post_sq_mean - post_mean * post_mean, 0.0, None))
    out = np.abs(pre_mean - post_mean) + 0.5 * np.abs(pre_std - post_std)
    out[~safe_valid] = 0.0
    out[~np.isfinite(out)] = 0.0
    return out


def _fill_invalid(arr: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    out = arr.astype(np.float32).copy()
    fill_value = float(np.median(out[valid_mask])) if valid_mask.any() else 0.0
    out[~valid_mask] = fill_value
    out[~np.isfinite(out)] = fill_value
    return out


def _box_filter(arr: np.ndarray, ksize: int) -> np.ndarray:
    pad = int(ksize // 2)
    padded = np.pad(arr, ((pad, pad), (pad, pad)), mode="reflect")
    out = np.zeros_like(arr, dtype=np.float32)
    area = float(ksize * ksize)
    for y in range(arr.shape[0]):
        for x in range(arr.shape[1]):
            out[y, x] = float(padded[y : y + ksize, x : x + ksize].sum() / area)
    return out
