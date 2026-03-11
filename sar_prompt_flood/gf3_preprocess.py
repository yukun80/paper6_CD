"""GF3_Henan 双时相大图预处理与切片。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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


def derive_change_products(pre: np.ndarray, post: np.ndarray, valid_mask: np.ndarray, clip_percentiles=(1.0, 99.0)) -> Dict[str, np.ndarray]:
    """从单极化 Pre/Post 影像构造变化先验。"""
    pre_norm = robust_unit(pre, valid_mask, clip_percentiles[0], clip_percentiles[1])
    post_norm = robust_unit(post, valid_mask, clip_percentiles[0], clip_percentiles[1])
    diff = post_norm - pre_norm
    neg_diff = np.clip(pre_norm - post_norm, a_min=0.0, a_max=None)
    log_ratio_like = safe_log_ratio(pre, post, valid_mask)
    log_ratio_like = robust_unit(-log_ratio_like, valid_mask, clip_percentiles[0], clip_percentiles[1])
    change_score = np.clip(0.75 * robust_unit(neg_diff, valid_mask, clip_percentiles[0], clip_percentiles[1]) + 0.25 * log_ratio_like, 0.0, 1.0)

    pre_norm[~valid_mask] = 0.0
    post_norm[~valid_mask] = 0.0
    diff[~valid_mask] = 0.0
    change_score[~valid_mask] = 0.0
    log_ratio_like[~valid_mask] = 0.0
    return {
        "pre_norm": pre_norm.astype(np.float32),
        "post_norm": post_norm.astype(np.float32),
        "diff": diff.astype(np.float32),
        "change_score": change_score.astype(np.float32),
        "log_ratio_like": log_ratio_like.astype(np.float32),
    }


def generate_windows(height: int, width: int, tile_size: int, overlap: int) -> List[Tuple[int, int, int, int]]:
    """生成固定重叠滑窗。"""
    stride = max(tile_size - overlap, 1)
    row_starts = list(range(0, max(height - tile_size, 0) + 1, stride))
    col_starts = list(range(0, max(width - tile_size, 0) + 1, stride))
    if not row_starts or row_starts[-1] != height - tile_size:
        row_starts.append(max(height - tile_size, 0))
    if not col_starts or col_starts[-1] != width - tile_size:
        col_starts.append(max(width - tile_size, 0))

    windows = []
    for top in row_starts:
        for left in col_starts:
            h = min(tile_size, height - top)
            w = min(tile_size, width - left)
            windows.append((top, left, h, w))
    return windows


def prepare_gf3_pair(
    pre_path: str,
    post_path: str,
    out_root: str,
    tile_size: int = 512,
    overlap: int = 128,
    min_valid_ratio: float = 0.6,
    clip_percentiles=(1.0, 99.0),
    max_tiles: int = -1,
) -> Dict:
    """对齐 GF3 双时相影像、构造先验并切 tile。"""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import Affine
    from rasterio.warp import reproject

    out_root = Path(out_root)
    aligned_dir = out_root / "aligned"
    priors_dir = out_root / "priors"
    tiles_dir = out_root / "tiles"
    aligned_dir.mkdir(parents=True, exist_ok=True)
    priors_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(pre_path) as pre_ds:
        pre = pre_ds.read(1).astype(np.float32)
        profile = pre_ds.profile.copy()
        ref_transform = pre_ds.transform
        ref_crs = pre_ds.crs
        ref_height = pre_ds.height
        ref_width = pre_ds.width
        ref_nodata = pre_ds.nodata
        ref_bounds = pre_ds.bounds

    with rasterio.open(post_path) as post_ds:
        post_raw = post_ds.read(1).astype(np.float32)
        post_aligned = np.full((ref_height, ref_width), fill_value=np.nan, dtype=np.float32)
        reproject(
            source=post_raw,
            destination=post_aligned,
            src_transform=post_ds.transform,
            src_crs=post_ds.crs,
            src_nodata=post_ds.nodata,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    pre = pre.copy()
    if ref_nodata is not None:
        pre[pre == ref_nodata] = np.nan
    valid_mask = np.isfinite(pre) & np.isfinite(post_aligned)
    products = derive_change_products(pre, post_aligned, valid_mask, clip_percentiles=clip_percentiles)

    save_single_band_geotiff(aligned_dir / "pre_aligned.tif", np.where(valid_mask, pre, np.nan), profile, ref_transform, ref_crs, nodata=np.nan)
    save_single_band_geotiff(aligned_dir / "post_aligned.tif", np.where(valid_mask, post_aligned, np.nan), profile, ref_transform, ref_crs, nodata=np.nan)
    save_single_band_geotiff(priors_dir / "diff.tif", products["diff"], profile, ref_transform, ref_crs, nodata=0.0)
    save_single_band_geotiff(priors_dir / "change_score.tif", products["change_score"], profile, ref_transform, ref_crs, nodata=0.0)
    save_single_band_geotiff(priors_dir / "log_ratio_like.tif", products["log_ratio_like"], profile, ref_transform, ref_crs, nodata=0.0)
    save_single_band_geotiff(priors_dir / "valid_mask.tif", valid_mask.astype(np.uint8), profile, ref_transform, ref_crs, nodata=0)

    manifest_tiles = []
    for idx, (top, left, h, w) in enumerate(generate_windows(ref_height, ref_width, tile_size, overlap)):
        if int(max_tiles) > 0 and len(manifest_tiles) >= int(max_tiles):
            break
        tile_valid = valid_mask[top : top + h, left : left + w]
        valid_ratio = float(tile_valid.mean())
        if valid_ratio < float(min_valid_ratio):
            continue
        tile_id = f"tile_{idx:06d}"
        tile_path = tiles_dir / f"{tile_id}.npz"
        pre_tile = products["pre_norm"][top : top + h, left : left + w]
        post_tile = products["post_norm"][top : top + h, left : left + w]
        diff_tile = products["diff"][top : top + h, left : left + w]
        score_tile = products["change_score"][top : top + h, left : left + w]
        lr_tile = products["log_ratio_like"][top : top + h, left : left + w]
        np.savez_compressed(
            tile_path,
            pre=pre_tile.astype(np.float32),
            post=post_tile.astype(np.float32),
            diff=diff_tile.astype(np.float32),
            change_score=score_tile.astype(np.float32),
            log_ratio_like=lr_tile.astype(np.float32),
            valid_mask=tile_valid.astype(np.uint8),
        )
        manifest_tiles.append(
            {
                "tile_id": tile_id,
                "npz": str(tile_path.relative_to(out_root)),
                "row_off": top,
                "col_off": left,
                "height": h,
                "width": w,
                "valid_ratio": valid_ratio,
                "low_confidence": bool(float(score_tile.max()) < 0.18),
            }
        )

    manifest = {
        "pair_name": "GF3_Henan",
        "pre_path": str(Path(pre_path).resolve()),
        "post_path": str(Path(post_path).resolve()),
        "reference": {
            "height": ref_height,
            "width": ref_width,
            "crs": str(ref_crs),
            "transform": list(ref_transform) if isinstance(ref_transform, Affine) else list(ref_transform),
            "bounds": [float(ref_bounds.left), float(ref_bounds.bottom), float(ref_bounds.right), float(ref_bounds.top)],
        },
        "tiles": manifest_tiles,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def save_single_band_geotiff(path: Path, arr: np.ndarray, profile: Dict, transform, crs, nodata) -> None:
    """写出单波段 GeoTIFF。"""
    import rasterio

    out_profile = profile.copy()
    out_profile.update(
        dtype=str(arr.dtype),
        count=1,
        transform=transform,
        crs=crs,
        compress="deflate",
        predictor=2,
        nodata=nodata,
        height=arr.shape[0],
        width=arr.shape[1],
    )
    with rasterio.open(path, "w", **out_profile) as ds:
        ds.write(arr, 1)
