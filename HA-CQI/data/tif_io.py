from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import rasterio

if TYPE_CHECKING:
    import torch


TIFF_SUFFIXES = {".tif", ".tiff"}


def is_tiff_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() in TIFF_SUFFIXES


def build_valid_mask(arr: np.ndarray, nodata: float | int | None) -> np.ndarray:
    """为 SAR tif 构造有效像素掩膜。"""
    valid = np.isfinite(arr)
    if nodata is not None:
        valid &= arr != nodata
    return valid


def stretch_sar_array(
    arr: np.ndarray,
    valid_mask: np.ndarray,
    low: float = 2.0,
    high: float = 98.0,
) -> np.ndarray:
    """按百分位稳定拉伸到 [0, 1]，便于后续增强与归一化。"""
    out = np.zeros(arr.shape, dtype=np.float32)
    if not np.any(valid_mask):
        return out

    values = arr[valid_mask].astype(np.float32, copy=False)
    lo = float(np.percentile(values, low))
    hi = float(np.percentile(values, high))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(values.min())
        hi = float(values.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return out

    scaled = (arr[valid_mask].astype(np.float32, copy=False) - lo) / (hi - lo)
    out[valid_mask] = np.clip(scaled, 0.0, 1.0)
    return out


def read_sar_tif(
    path: str | Path,
    stretch_low: float = 2.0,
    stretch_high: float = 98.0,
    replicate_channels: int = 3,
) -> torch.Tensor:
    """读取单波段 SAR tif，并复制为模型所需的 3 通道张量。"""
    import torch

    with rasterio.open(path) as ds:
        arr = ds.read(1).astype(np.float32, copy=False)
        valid_mask = build_valid_mask(arr, ds.nodata)

    stretched = stretch_sar_array(arr, valid_mask, low=stretch_low, high=stretch_high)
    stacked = np.repeat(stretched[None, ...], replicate_channels, axis=0)
    return torch.from_numpy(stacked.copy())


def read_binary_label_tif_with_valid_mask(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """读取 0/1 标签并返回有效掩膜；兼容历史 nodata=3。"""
    with rasterio.open(path) as ds:
        arr = ds.read(1)
        valid = build_valid_mask(arr, ds.nodata)
    # 历史 GF-3 标签使用 3 表示无效区，但部分文件未写入 nodata metadata。
    valid &= arr != 3
    label = ((arr > 0) & valid).astype(np.uint8)
    return label, valid


def read_binary_label_tif(path: str | Path) -> np.ndarray:
    """读取 tif 标签并安全转成 0/1；无效像素返回背景值 0。"""
    label, _ = read_binary_label_tif_with_valid_mask(path)
    return label
