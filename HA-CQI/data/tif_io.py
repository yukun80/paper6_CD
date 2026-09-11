from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import MaskFlags
from rasterio.windows import Window

if TYPE_CHECKING:
    import torch


TIFF_SUFFIXES = {".tif", ".tiff"}
LABEL_FOREGROUND = 255
LABEL_NODATA = 254


def is_tiff_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() in TIFF_SUFFIXES


def build_valid_mask(arr: np.ndarray, nodata: float | int | None) -> np.ndarray:
    """为 SAR tif 构造有效像素掩膜。"""
    valid = np.isfinite(arr)
    if nodata is not None:
        valid &= arr != nodata
    return valid


def read_raster_valid_mask(
    dataset: rasterio.io.DatasetReader,
    arr: np.ndarray,
    window: Window | None = None,
) -> np.ndarray:
    """影像覆盖区同时遵守数值 NoData、非有限值和 GDAL 掩膜。"""
    return build_valid_mask(arr, dataset.nodata) & (dataset.read_masks(1, window=window) > 0)


def read_label_mask(
    dataset: rasterio.io.DatasetReader, window: Window | None = None,
) -> np.ndarray | None:
    """GT 的零值是背景；忽略仅由 nodata=0 派生的掩膜，保留独立掩膜。"""
    if dataset.nodata == 0 and set(dataset.mask_flag_enums[0]) == {MaskFlags.nodata}:
        return None
    return dataset.read_masks(1, window=window) > 0


def decode_binary_label(
    arr: np.ndarray,
    nodata: float | int | None = None,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """兼容旧洪水值 1/255；未知值归背景，返回的源有效性仅用于审计。"""
    arr = np.asarray(arr)
    valid = np.isfinite(arr) & ((arr == 0) | (arr == 1) | (arr == 255))
    if nodata is not None and nodata != 0:
        valid &= arr != nodata
    if valid_mask is not None:
        if valid_mask.shape != arr.shape:
            raise ValueError(f"Label/mask shape mismatch: {arr.shape} != {valid_mask.shape}")
        valid &= np.asarray(valid_mask, dtype=bool)
    label = (((arr == 1) | (arr == 255)) & valid).astype(np.uint8)
    return label, valid


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
        valid_mask = read_raster_valid_mask(ds, arr)

    stretched = stretch_sar_array(arr, valid_mask, low=stretch_low, high=stretch_high)
    stacked = np.repeat(stretched[None, ...], replicate_channels, axis=0)
    return torch.from_numpy(stacked.copy())


def read_binary_label_tif_with_valid_mask(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """读取 GT 为 0/1；源未知区归背景，源有效性仅用于审计。"""
    with rasterio.open(path) as ds:
        arr = ds.read(1)
        return decode_binary_label(arr, ds.nodata, read_label_mask(ds))


def read_binary_label_tif(path: str | Path) -> np.ndarray:
    """读取 tif 标签并安全转成 0/1；无效像素返回背景值 0。"""
    label, _ = read_binary_label_tif_with_valid_mask(path)
    return label


def read_binary_label_with_valid_mask(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """统一 PNG/TIFF 标签语义；普通黑色像素不作为 NoData。"""
    if is_tiff_path(path):
        return read_binary_label_tif_with_valid_mask(path)
    with Image.open(path) as image:
        # 数值灰度保留原精度，避免意外值在 uint8 转换时变成合法前景。
        arr = np.asarray(image if image.mode in {"L", "I", "F", "I;16"} else image.convert("L"))
        mask = None
        if "A" in image.getbands() or "transparency" in image.info:
            mask = np.asarray(image.convert("RGBA").getchannel("A")) > 0
        return decode_binary_label(arr, valid_mask=mask)


def read_training_label(
    path: str | Path, image_paths: Iterable[str | Path] = (),
) -> np.ndarray:
    """按配对影像覆盖区归零 GT，Dataset、快照和构建器共用此监督规则。"""
    label, _ = read_binary_label_with_valid_mask(path)
    for image_path in image_paths:
        if not is_tiff_path(image_path):
            continue
        with rasterio.open(image_path) as dataset:
            arr = dataset.read(1)
            valid = read_raster_valid_mask(dataset, arr)
        if label.shape != valid.shape:
            raise ValueError(f"Label/image shape mismatch: {path} vs {image_path}")
        label[~valid] = 0
    return label
