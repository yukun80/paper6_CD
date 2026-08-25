"""将单波段整景 SAR tif 切成 ChangeDINO 推理所需的 PNG/TIF 切片。

python baselines/ChangeDINO-main/scripts/prepare_sar_scene_infer.py \
  --src-root datasets/USA_Brazos_River \
  --pre-image Pre_Brazos_ascending_20170805.tif \
  --post-image Post_Brazos_ascending_20170829.tif \
  --out-root datasets/USA_Brazos_River_CD_infer \
  --scene-tag usa_brazos_river \
  --tile-size 256 \
  --stride 128 \
  --stretch-low 2 \
  --stretch-high 98 \
  --min-valid-ratio 0.01 \
  --strict

python baselines/ChangeDINO-main/scripts/prepare_sar_scene_infer.py \
  --src-root datasets/USA_San_Jacinto \
  --pre-image Pre_SanJacinto_ascending_20170805.tif \
  --post-image Post_SanJacinto_ascending_20170829.tif \
  --out-root datasets/USA_San_Jacinto_CD_infer \
  --scene-tag usa_san_jacinto \
  --tile-size 256 \
  --stride 128 \
  --stretch-low 2 \
  --stretch-high 98 \
  --min-valid-ratio 0.01 \
  --strict

"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window


@dataclass(frozen=True)
class SceneMeta:
    """记录整景配对影像的关键元数据，避免后续重复打开文件判断。"""

    pre_path: Path
    post_path: Path
    width: int
    height: int
    crs: str | None
    transform: str
    nodata_pre: float | int | None
    nodata_post: float | int | None
    dtype_pre: str
    dtype_post: str


@dataclass(frozen=True)
class StretchConfig:
    """记录单路 PNG 拉伸配置，便于 pre/post 分开处理。"""

    mode: str
    low: float
    high: float
    value_min: float | None
    value_max: float | None


def build_parser(
    description: str = "Prepare SAR scene tiles for ChangeDINO inference",
    defaults: dict[str, object] | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description)
    parser.add_argument("--src-root", type=Path, default=Path("datasets/SAR_Scene"))
    parser.add_argument("--pre-image", type=str, default="pre.tif")
    parser.add_argument("--post-image", type=str, default="post.tif")
    parser.add_argument("--out-root", type=Path, default=Path("datasets/SAR_Scene_CD_infer"))
    parser.add_argument("--scene-tag", type=str, default="sar_scene", help="切片 id 前缀。")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument(
        "--stretch-mode",
        type=str,
        choices=["percentile", "value"],
        default="percentile",
        help="PNG 切片拉伸方式：按分位数或按固定数值窗口裁剪。",
    )
    parser.add_argument("--stretch-low", type=float, default=2.0)
    parser.add_argument("--stretch-high", type=float, default=98.0)
    parser.add_argument("--value-min", type=float, default=None, help="固定数值拉伸下限，仅 stretch-mode=value 时生效。")
    parser.add_argument("--value-max", type=float, default=None, help="固定数值拉伸上限，仅 stretch-mode=value 时生效。")
    parser.add_argument(
        "--pre-stretch-mode",
        type=str,
        choices=["percentile", "value"],
        default=None,
        help="可选：覆盖 pre-image 的 PNG 拉伸方式。",
    )
    parser.add_argument("--pre-stretch-low", type=float, default=None, help="可选：覆盖 pre-image 的百分位拉伸下限。")
    parser.add_argument("--pre-stretch-high", type=float, default=None, help="可选：覆盖 pre-image 的百分位拉伸上限。")
    parser.add_argument("--pre-value-min", type=float, default=None, help="可选：覆盖 pre-image 的固定数值拉伸下限。")
    parser.add_argument("--pre-value-max", type=float, default=None, help="可选：覆盖 pre-image 的固定数值拉伸上限。")
    parser.add_argument(
        "--post-stretch-mode",
        type=str,
        choices=["percentile", "value"],
        default=None,
        help="可选：覆盖 post-image 的 PNG 拉伸方式。",
    )
    parser.add_argument("--post-stretch-low", type=float, default=None, help="可选：覆盖 post-image 的百分位拉伸下限。")
    parser.add_argument("--post-stretch-high", type=float, default=None, help="可选：覆盖 post-image 的百分位拉伸上限。")
    parser.add_argument("--post-value-min", type=float, default=None, help="可选：覆盖 post-image 的固定数值拉伸下限。")
    parser.add_argument("--post-value-max", type=float, default=None, help="可选：覆盖 post-image 的固定数值拉伸上限。")
    parser.add_argument("--min-valid-ratio", type=float, default=0.01)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--strict", action="store_true")
    if defaults:
        parser.set_defaults(**defaults)
    return parser


def parse_args(
    argv: Sequence[str] | None = None,
    defaults: dict[str, object] | None = None,
    description: str = "Prepare SAR scene tiles for ChangeDINO inference",
) -> argparse.Namespace:
    return build_parser(description=description, defaults=defaults).parse_args(argv)


def validate_stretch_config(name: str, config: StretchConfig) -> None:
    """校验单路拉伸配置，避免 pre/post 混合覆盖时产生歧义。"""
    if config.mode == "percentile":
        if not 0.0 <= config.low < config.high <= 100.0:
            raise ValueError(f"{name}: percentile stretch requires 0 <= low < high <= 100")
        return

    if config.value_min is None or config.value_max is None:
        raise ValueError(f"{name}: value stretch requires value_min/value_max")
    if not np.isfinite(config.value_min) or not np.isfinite(config.value_max):
        raise ValueError(f"{name}: value_min/value_max must be finite")
    if config.value_min >= config.value_max:
        raise ValueError(f"{name}: value_min must be smaller than value_max")


def resolve_stretch_config(args: argparse.Namespace, target: str) -> StretchConfig:
    """将全局参数与 pre/post 专属参数合并为实际生效配置。"""
    if target not in {"pre", "post"}:
        raise ValueError(f"Unsupported stretch target: {target}")

    mode = getattr(args, f"{target}_stretch_mode")
    low = getattr(args, f"{target}_stretch_low")
    high = getattr(args, f"{target}_stretch_high")
    value_min = getattr(args, f"{target}_value_min")
    value_max = getattr(args, f"{target}_value_max")
    return StretchConfig(
        mode=args.stretch_mode if mode is None else mode,
        low=float(args.stretch_low if low is None else low),
        high=float(args.stretch_high if high is None else high),
        value_min=args.value_min if value_min is None else float(value_min),
        value_max=args.value_max if value_max is None else float(value_max),
    )


def ensure_args(args: argparse.Namespace) -> None:
    if not args.src_root.is_dir():
        raise FileNotFoundError(f"Source root not found: {args.src_root}")
    if args.tile_size <= 0 or args.stride <= 0:
        raise ValueError("--tile-size and --stride must be positive")
    if not 0.0 <= args.min_valid_ratio <= 1.0:
        raise ValueError("--min-valid-ratio must be within [0, 1]")
    validate_stretch_config(
        "default stretch",
        StretchConfig(
            mode=args.stretch_mode,
            low=float(args.stretch_low),
            high=float(args.stretch_high),
            value_min=args.value_min,
            value_max=args.value_max,
        ),
    )
    validate_stretch_config("pre stretch", resolve_stretch_config(args, "pre"))
    validate_stretch_config("post stretch", resolve_stretch_config(args, "post"))
    if not args.scene_tag:
        raise ValueError("--scene-tag must be non-empty")


def validate_scene(pre_path: Path, post_path: Path, strict: bool) -> SceneMeta:
    """校验灾前/灾后影像能否按像素一一配对。"""
    if not pre_path.is_file():
        raise FileNotFoundError(f"Missing pre image: {pre_path}")
    if not post_path.is_file():
        raise FileNotFoundError(f"Missing post image: {post_path}")

    with rasterio.open(pre_path) as ds_pre, rasterio.open(post_path) as ds_post:
        if ds_pre.count != 1 or ds_post.count != 1:
            raise ValueError("SAR scene preprocessing expects single-band tif inputs")
        if (ds_pre.width, ds_pre.height) != (ds_post.width, ds_post.height):
            raise ValueError("Pre/Post shape mismatch")
        if ds_pre.crs != ds_post.crs:
            raise ValueError("Pre/Post CRS mismatch")
        if ds_pre.transform != ds_post.transform:
            raise ValueError("Pre/Post geotransform mismatch")
        if strict and (ds_pre.dtypes[0] != "float32" or ds_post.dtypes[0] != "float32"):
            raise ValueError(
                f"Strict mode expects float32 tif, got {ds_pre.dtypes[0]} and {ds_post.dtypes[0]}"
            )

        return SceneMeta(
            pre_path=pre_path,
            post_path=post_path,
            width=ds_pre.width,
            height=ds_pre.height,
            crs=str(ds_pre.crs) if ds_pre.crs else None,
            transform=str(ds_pre.transform),
            nodata_pre=ds_pre.nodata,
            nodata_post=ds_post.nodata,
            dtype_pre=ds_pre.dtypes[0],
            dtype_post=ds_post.dtypes[0],
        )


def build_positions(length: int, tile_size: int, stride: int) -> list[int]:
    """生成覆盖整景边界的起始坐标，保证最后一行/列不会漏切。"""
    if length <= tile_size:
        return [0]

    positions = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return sorted(set(positions))


def iter_windows(height: int, width: int, tile_size: int, stride: int) -> list[Window]:
    row_positions = build_positions(height, tile_size, stride)
    col_positions = build_positions(width, tile_size, stride)
    windows: list[Window] = []
    for top in row_positions:
        for left in col_positions:
            windows.append(Window(col_off=left, row_off=top, width=tile_size, height=tile_size))
    return windows


def build_valid_mask(
    arr_pre: np.ndarray,
    nodata_pre: float | int | None,
    arr_post: np.ndarray,
    nodata_post: float | int | None,
) -> np.ndarray:
    valid = np.isfinite(arr_pre) & np.isfinite(arr_post)
    if nodata_pre is not None:
        valid &= np.not_equal(arr_pre, nodata_pre)
    if nodata_post is not None:
        valid &= np.not_equal(arr_post, nodata_post)
    return valid


def stretch_to_uint8(
    arr: np.ndarray,
    valid_mask: np.ndarray,
    config: StretchConfig,
) -> np.ndarray:
    """仅基于有效像素做拉伸，保证 nodata 不污染分位点统计。"""
    out = np.zeros(arr.shape, dtype=np.uint8)
    if not np.any(valid_mask):
        return out

    if config.mode == "value":
        if config.value_min is None or config.value_max is None:
            raise ValueError("value_min/value_max are required for fixed-value stretching")
        lo = float(config.value_min)
        hi = float(config.value_max)
    else:
        values = arr[valid_mask].astype(np.float32, copy=False)
        lo = float(np.percentile(values, config.low))
        hi = float(np.percentile(values, config.high))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(values.min())
            hi = float(values.max())
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return out

    scaled = np.zeros(arr.shape, dtype=np.float32)
    scaled[valid_mask] = np.clip((arr[valid_mask].astype(np.float32, copy=False) - lo) / (hi - lo), 0.0, 1.0)
    out[valid_mask] = np.rint(scaled[valid_mask] * 255.0).astype(np.uint8)
    return out


def clean_output_root(out_root: Path, overwrite: bool, dry_run: bool) -> None:
    if not out_root.exists():
        return
    if not overwrite or dry_run:
        return
    shutil.rmtree(out_root)


def prepare_dirs(out_root: Path, dry_run: bool) -> None:
    for subdir in ("test/A", "test/B", "test/A_tif", "test/B_tif", "test/valid_mask"):
        if not dry_run:
            (out_root / subdir).mkdir(parents=True, exist_ok=True)


def write_tif(
    src_ds: rasterio.io.DatasetReader,
    arr: np.ndarray,
    out_path: Path,
    window: Window,
    dtype: str,
    nodata: float | int | None,
    dry_run: bool,
) -> None:
    if dry_run:
        return
    profile = src_ds.profile.copy()
    profile.update(
        {
            "driver": "GTiff",
            "height": int(window.height),
            "width": int(window.width),
            "count": 1,
            "dtype": dtype,
            "transform": rasterio.windows.transform(window, src_ds.transform),
            "compress": "LZW",
            "nodata": nodata,
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as out_ds:
        out_ds.write(arr, 1)


def write_png_rgb(arr: np.ndarray, out_path: Path, dry_run: bool) -> None:
    if dry_run:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.repeat(arr[:, :, None], 3, axis=2)
    Image.fromarray(rgb, mode="RGB").save(out_path)


def write_outputs(
    args: argparse.Namespace,
    scene: SceneMeta,
    windows: list[Window],
    pre_stretch: StretchConfig,
    post_stretch: StretchConfig,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """同时导出 PNG/TIF/valid_mask，并记录 manifest 行。"""
    manifest_rows: list[dict[str, object]] = []
    skipped = 0

    with rasterio.open(scene.pre_path) as ds_pre, rasterio.open(scene.post_path) as ds_post:
        for index, window in enumerate(windows):
            top = int(window.row_off)
            left = int(window.col_off)
            arr_pre = ds_pre.read(1, window=window).astype(np.float32, copy=False)
            arr_post = ds_post.read(1, window=window).astype(np.float32, copy=False)
            valid_mask = build_valid_mask(arr_pre, ds_pre.nodata, arr_post, ds_post.nodata)
            valid_ratio = float(valid_mask.mean())
            if valid_ratio < args.min_valid_ratio:
                skipped += 1
                continue

            tile_id = f"{args.scene_tag}_r{top:05d}_c{left:05d}"
            pre_png_rel = Path("test/A") / f"{tile_id}.png"
            post_png_rel = Path("test/B") / f"{tile_id}.png"
            pre_tif_rel = Path("test/A_tif") / f"{tile_id}.tif"
            post_tif_rel = Path("test/B_tif") / f"{tile_id}.tif"
            valid_tif_rel = Path("test/valid_mask") / f"{tile_id}.tif"

            pre_png = stretch_to_uint8(arr_pre, valid_mask, pre_stretch)
            post_png = stretch_to_uint8(arr_post, valid_mask, post_stretch)

            write_png_rgb(pre_png, args.out_root / pre_png_rel, args.dry_run)
            write_png_rgb(post_png, args.out_root / post_png_rel, args.dry_run)
            write_tif(ds_pre, arr_pre, args.out_root / pre_tif_rel, window, "float32", ds_pre.nodata, args.dry_run)
            write_tif(ds_post, arr_post, args.out_root / post_tif_rel, window, "float32", ds_post.nodata, args.dry_run)
            write_tif(
                ds_pre,
                valid_mask.astype(np.uint8),
                args.out_root / valid_tif_rel,
                window,
                "uint8",
                0,
                args.dry_run,
            )

            manifest_rows.append(
                {
                    "tile_id": tile_id,
                    "split": "test",
                    "top": top,
                    "left": left,
                    "height": int(window.height),
                    "width": int(window.width),
                    "valid_ratio": f"{valid_ratio:.6f}",
                    "a_png": str(pre_png_rel),
                    "b_png": str(post_png_rel),
                    "a_tif": str(pre_tif_rel),
                    "b_tif": str(post_tif_rel),
                    "valid_mask": str(valid_tif_rel),
                    "tile_index": index,
                }
            )

    return manifest_rows, {"skipped_tiles": skipped}


def write_manifest(out_root: Path, manifest_rows: list[dict[str, object]], dry_run: bool) -> None:
    if dry_run:
        return
    fieldnames = [
        "tile_id",
        "split",
        "top",
        "left",
        "height",
        "width",
        "valid_ratio",
        "a_png",
        "b_png",
        "a_tif",
        "b_tif",
        "valid_mask",
        "tile_index",
    ]
    with (out_root / "tile_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)


def write_report(
    args: argparse.Namespace,
    scene: SceneMeta,
    candidate_tiles: int,
    kept_tiles: int,
    skipped_tiles: int,
    manifest_rows: list[dict[str, object]],
    pre_stretch: StretchConfig,
    post_stretch: StretchConfig,
) -> None:
    payload = {
        "source": {
            "src_root": str(args.src_root),
            "pre_image": str(scene.pre_path.resolve()),
            "post_image": str(scene.post_path.resolve()),
            "width": scene.width,
            "height": scene.height,
            "crs": scene.crs,
            "transform": scene.transform,
            "dtype_pre": scene.dtype_pre,
            "dtype_post": scene.dtype_post,
            "nodata_pre": scene.nodata_pre,
            "nodata_post": scene.nodata_post,
        },
        "params": {
            "scene_tag": args.scene_tag,
            "tile_size": args.tile_size,
            "stride": args.stride,
            "stretch_mode": args.stretch_mode,
            "stretch_low": args.stretch_low,
            "stretch_high": args.stretch_high,
            "value_min": args.value_min,
            "value_max": args.value_max,
            "pre_stretch_mode": args.pre_stretch_mode,
            "pre_stretch_low": args.pre_stretch_low,
            "pre_stretch_high": args.pre_stretch_high,
            "pre_value_min": args.pre_value_min,
            "pre_value_max": args.pre_value_max,
            "post_stretch_mode": args.post_stretch_mode,
            "post_stretch_low": args.post_stretch_low,
            "post_stretch_high": args.post_stretch_high,
            "post_value_min": args.post_value_min,
            "post_value_max": args.post_value_max,
            "min_valid_ratio": args.min_valid_ratio,
            "strict": args.strict,
            "dry_run": args.dry_run,
        },
        "effective_stretch": {
            "pre": {
                "mode": pre_stretch.mode,
                "low": pre_stretch.low,
                "high": pre_stretch.high,
                "value_min": pre_stretch.value_min,
                "value_max": pre_stretch.value_max,
            },
            "post": {
                "mode": post_stretch.mode,
                "low": post_stretch.low,
                "high": post_stretch.high,
                "value_min": post_stretch.value_min,
                "value_max": post_stretch.value_max,
            },
        },
        "tile_counts": {
            "candidate_tiles": candidate_tiles,
            "kept_tiles": kept_tiles,
            "skipped_tiles": skipped_tiles,
        },
        "examples": [row["tile_id"] for row in manifest_rows[:5]],
    }
    if args.dry_run:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    (args.out_root / "prepare_report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    defaults: dict[str, object] | None = None,
    description: str = "Prepare SAR scene tiles for ChangeDINO inference",
) -> None:
    args = parse_args(argv=argv, defaults=defaults, description=description)
    args.src_root = Path(args.src_root)
    args.out_root = Path(args.out_root)
    ensure_args(args)
    pre_stretch = resolve_stretch_config(args, "pre")
    post_stretch = resolve_stretch_config(args, "post")

    pre_path = args.src_root / args.pre_image
    post_path = args.src_root / args.post_image
    scene = validate_scene(pre_path, post_path, args.strict)
    windows = iter_windows(scene.height, scene.width, args.tile_size, args.stride)

    clean_output_root(args.out_root, args.overwrite, args.dry_run)
    prepare_dirs(args.out_root, args.dry_run)
    manifest_rows, counters = write_outputs(args, scene, windows, pre_stretch, post_stretch)
    write_manifest(args.out_root, manifest_rows, args.dry_run)
    write_report(
        args,
        scene,
        candidate_tiles=len(windows),
        kept_tiles=len(manifest_rows),
        skipped_tiles=counters["skipped_tiles"],
        manifest_rows=manifest_rows,
        pre_stretch=pre_stretch,
        post_stretch=post_stretch,
    )

    print(
        f"[DONE] candidate={len(windows)} kept={len(manifest_rows)} "
        f"skipped={counters['skipped_tiles']}"
    )


if __name__ == "__main__":
    main()
