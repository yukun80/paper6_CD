"""栅格读写、配置解析与日志工具。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import rasterio
import yaml
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject
from rasterio.windows import Window


RasterBundle = Dict[str, Any]


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def resolve_path(path: str | Path, base_dir: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (Path(base_dir) / candidate).resolve()


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def setup_logger(output_dir: str | Path, verbose: bool = True) -> logging.Logger:
    ensure_dir(output_dir)
    logger_name = f"hsba_flood_{Path(output_dir).resolve()}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(Path(output_dir) / "run.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)
    return logger


def read_raster(path: str | Path) -> RasterBundle:
    path = Path(path)
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype(np.float32)
        profile = ds.profile.copy()
        transform = ds.transform
        crs = ds.crs
        nodata = ds.nodata
        bounds = ds.bounds
    return {
        "path": str(path),
        "array": arr,
        "profile": profile,
        "transform": transform,
        "crs": crs,
        "nodata": nodata,
        "bounds": bounds,
    }


def grids_match(lhs: RasterBundle, rhs: RasterBundle) -> bool:
    return (
        lhs["array"].shape == rhs["array"].shape
        and lhs["crs"] == rhs["crs"]
        and tuple(lhs["transform"]) == tuple(rhs["transform"])
    )


def align_raster_to_reference(
    source: RasterBundle,
    reference: RasterBundle,
    resampling: Resampling = Resampling.bilinear,
) -> RasterBundle:
    """将 source 重投影到 reference 网格。"""

    dst = np.full(reference["array"].shape, np.nan, dtype=np.float32)
    src_nodata = source["nodata"]
    reproject(
        source=source["array"],
        destination=dst,
        src_transform=source["transform"],
        src_crs=source["crs"],
        src_nodata=src_nodata,
        dst_transform=reference["transform"],
        dst_crs=reference["crs"],
        dst_nodata=np.nan,
        resampling=resampling,
    )
    profile = reference["profile"].copy()
    profile.update(dtype="float32", count=1, nodata=np.nan)
    return {
        "path": source["path"],
        "array": dst,
        "profile": profile,
        "transform": reference["transform"],
        "crs": reference["crs"],
        "nodata": np.nan,
        "bounds": reference["bounds"],
        "aligned_from": source["path"],
    }


def write_float_raster(
    path: str | Path,
    array: np.ndarray,
    reference_profile: Dict[str, Any],
    nodata: Optional[float] = np.nan,
) -> None:
    profile = _prepare_write_profile(reference_profile, dtype="float32", nodata=nodata, predictor=3)
    profile.update(count=1)
    if nodata is not None:
        profile.update(nodata=nodata)
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(array.astype(np.float32), 1)


def _iter_windows(height: int, width: int, block_size: int) -> Tuple[Window, slice, slice]:
    for row_start in range(0, height, block_size):
        row_end = min(row_start + block_size, height)
        row_slice = slice(row_start, row_end)
        for col_start in range(0, width, block_size):
            col_end = min(col_start + block_size, width)
            col_slice = slice(col_start, col_end)
            yield Window(col_start, row_start, col_end - col_start, row_end - row_start), row_slice, col_slice


def write_validated_float_raster(
    path: str | Path,
    array: np.ndarray,
    reference_profile: Dict[str, Any],
    valid_mask: np.ndarray,
    nodata: Optional[float] = np.nan,
    block_size: int = 1024,
) -> None:
    """分块写出带 valid mask 的浮点影像，避免额外整图副本。"""

    profile = _prepare_write_profile(reference_profile, dtype="float32", nodata=nodata, predictor=3)
    profile.update(count=1)
    with rasterio.open(path, "w", **profile) as ds:
        for window, row_slice, col_slice in _iter_windows(array.shape[0], array.shape[1], block_size):
            block_valid = valid_mask[row_slice, col_slice]
            block = np.full(block_valid.shape, nodata, dtype=np.float32)
            block[block_valid] = array[row_slice, col_slice][block_valid].astype(np.float32, copy=False)
            ds.write(block, 1, window=window)


def write_validated_difference_raster(
    path: str | Path,
    left: np.ndarray,
    right: np.ndarray,
    reference_profile: Dict[str, Any],
    valid_mask: np.ndarray,
    nodata: Optional[float] = np.nan,
    block_size: int = 1024,
) -> None:
    """分块写出差值图，避免构造整幅 XC 副本。"""

    profile = _prepare_write_profile(reference_profile, dtype="float32", nodata=nodata, predictor=3)
    profile.update(count=1)
    with rasterio.open(path, "w", **profile) as ds:
        for window, row_slice, col_slice in _iter_windows(left.shape[0], left.shape[1], block_size):
            block_valid = valid_mask[row_slice, col_slice]
            block = np.full(block_valid.shape, nodata, dtype=np.float32)
            if np.any(block_valid):
                block_left = left[row_slice, col_slice]
                block_right = right[row_slice, col_slice]
                block[block_valid] = (block_left[block_valid] - block_right[block_valid]).astype(np.float32, copy=False)
            ds.write(block, 1, window=window)


def write_mask_raster(
    path: str | Path,
    mask: np.ndarray,
    reference_profile: Dict[str, Any],
    nodata: int = 255,
    valid_mask: Optional[np.ndarray] = None,
) -> None:
    data = np.asarray(mask).astype(np.uint8)
    out = np.zeros_like(data, dtype=np.uint8)
    out[data > 0] = 1
    if valid_mask is not None:
        out[~valid_mask] = np.uint8(nodata)
    profile = _prepare_write_profile(reference_profile, dtype="uint8", nodata=nodata)
    profile.update(count=1)
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(out, 1)


def _prepare_write_profile(
    reference_profile: Dict[str, Any],
    dtype: str,
    nodata: Optional[float | int],
    predictor: Optional[int] = None,
) -> Dict[str, Any]:
    """统一清理写出 profile，避免继承源影像的非法 tile 配置。"""

    profile = reference_profile.copy()
    profile.update(dtype=dtype, compress="deflate", tiled=True)
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)

    width = int(profile["width"])
    height = int(profile["height"])
    blockxsize = max(16, min(512, width))
    blockysize = max(16, min(512, height))
    blockxsize = ((blockxsize + 15) // 16) * 16
    blockysize = ((blockysize + 15) // 16) * 16
    blockxsize = min(blockxsize, ((width + 15) // 16) * 16)
    blockysize = min(blockysize, ((height + 15) // 16) * 16)
    profile.update(blockxsize=blockxsize, blockysize=blockysize)

    if nodata is not None:
        profile.update(nodata=nodata)
    if predictor is not None:
        profile.update(predictor=predictor)
    return profile


def serializable_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in profile.items():
        if isinstance(value, Affine):
            result[key] = list(value)
        elif key == "crs" and value is not None:
            result[key] = str(value)
        else:
            result[key] = value
    return result


def build_run_metadata(bundle: RasterBundle) -> Dict[str, Any]:
    return {
        "path": bundle["path"],
        "shape": list(bundle["array"].shape),
        "nodata": None if bundle["nodata"] is None else float(bundle["nodata"]),
        "transform": list(bundle["transform"]),
        "crs": None if bundle["crs"] is None else str(bundle["crs"]),
        "bounds": [float(v) for v in bundle["bounds"]],
        "profile": serializable_profile(bundle["profile"]),
    }
