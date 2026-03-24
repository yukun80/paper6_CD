"""NoData 与有效像元掩膜工具。"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def build_valid_mask(array: np.ndarray, nodata: Optional[float], nodata_override: Optional[float] = None) -> np.ndarray:
    """构造单景有效像元掩膜。"""

    valid = np.isfinite(array)
    if nodata is not None and np.isfinite(nodata):
        valid &= array != np.float32(nodata)
    if nodata_override is not None and np.isfinite(nodata_override):
        valid &= array != np.float32(nodata_override)
    return valid


def combine_valid_masks(*masks: np.ndarray) -> np.ndarray:
    valid = np.ones_like(masks[0], dtype=bool)
    for mask in masks:
        valid &= mask.astype(bool)
    return valid


def masked_copy(array: np.ndarray, valid_mask: np.ndarray, fill_value: float = np.nan) -> np.ndarray:
    out = np.full(array.shape, fill_value, dtype=np.float32)
    out[valid_mask] = array[valid_mask].astype(np.float32)
    return out


def count_valid_ratio(valid_mask: np.ndarray) -> Tuple[int, int, float]:
    valid_pixels = int(valid_mask.sum())
    total_pixels = int(valid_mask.size)
    ratio = float(valid_pixels / total_pixels) if total_pixels > 0 else 0.0
    return valid_pixels, total_pixels, ratio
