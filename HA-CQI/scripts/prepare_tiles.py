from __future__ import annotations
import argparse
import csv
import json
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.scene_stretch import (  # noqa: E402
    StretchConfig,
    fit_scene_stretch,
    read_scene_window,
    stretch_to_uint8,
    validate_stretch,
)

PREPARE_SCHEMA_VERSION = 1
# 目录创建与 manifest 写出共用同一布局，避免路径不一致。
TILE_SUBDIR = Path("test")
SCENE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")

"""
将同网格灾前/灾后 SAR 整景切为 HA-CQI 推理输入，仅使用命令行配置。
python HA-CQI/scripts/prepare_tiles.py \
  --pre-image datasets/LT1_Guangxi/LT_Guangxi_pre.tif \
  --post-image datasets/LT1_Guangxi/LT_Guangxi_post.tif \
  --output-dir datasets/LT1_Guangxi_CD_infer \
  --scene-id LT1_Guangxi \
  --overwrite
"""


@dataclass(frozen=True)
class PrepareOptions:
    """切片阶段经 CLI/YAML 合并后的确定性参数。"""

    pre_image: Path
    post_image: Path
    output_dir: Path
    scene_id: str
    tile_size: int = 256
    stride: int = 128
    min_valid_ratio: float = 0.01
    stretch: StretchConfig = StretchConfig()
    dry_run: bool = False
    overwrite: bool = False



def validate_options(options: PrepareOptions) -> None:
    if not options.pre_image.is_file():
        raise FileNotFoundError(f"Pre-event SAR image not found: {options.pre_image}")
    if not options.post_image.is_file():
        raise FileNotFoundError(f"Post-event SAR image not found: {options.post_image}")
    if not SCENE_ID_PATTERN.fullmatch(options.scene_id):
        raise ValueError("scene_id may only contain letters, digits, '_' and '-'")
    if options.tile_size != 256:
        raise ValueError("This checkpoint has a fixed 256 x 256 model input")
    if not 1 <= options.stride <= options.tile_size:
        raise ValueError("stride must be within [1, tile_size]")
    if not 0.0 <= options.min_valid_ratio <= 1.0:
        raise ValueError("min_valid_ratio must be within [0, 1]")
    resolved_output = options.output_dir.resolve()
    for input_path in (options.pre_image.resolve(), options.post_image.resolve()):
        if resolved_output in input_path.parents:
            raise ValueError(
                f"Output directory may not contain an input image: {resolved_output}"
            )
    validate_stretch(options.stretch)



def build_positions(length: int, tile_size: int, stride: int) -> list[int]:
    """生成包含末端边界的滑窗起点。"""
    if length < tile_size:
        raise ValueError(
            f"Scene dimension {length} is smaller than the fixed tile size {tile_size}"
        )
    positions = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions



def iter_windows(height: int, width: int, tile_size: int, stride: int) -> list[Window]:
    return [
        Window(left, top, tile_size, tile_size)
        for top in build_positions(height, tile_size, stride)
        for left in build_positions(width, tile_size, stride)
    ]



def _safe_nodata(value: float | int | None) -> float | int | None:
    if value is None:
        return None
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None



def _source_metadata(dataset: rasterio.io.DatasetReader) -> dict[str, Any]:
    crs = dataset.crs
    return {
        "width": int(dataset.width),
        "height": int(dataset.height),
        "crs_epsg": crs.to_epsg() if crs is not None else None,
        "crs_wkt": crs.to_wkt() if crs is not None else None,
        "transform": [
            float(dataset.transform.a),
            float(dataset.transform.b),
            float(dataset.transform.c),
            float(dataset.transform.d),
            float(dataset.transform.e),
            float(dataset.transform.f),
        ],
        "dtype": dataset.dtypes[0],
        "nodata": _safe_nodata(dataset.nodata),
    }



def validate_scene_pair(
    pre_image: Path,
    post_image: Path,
    tile_size: int,
) -> dict[str, Any]:
    """拒绝隐式重投影、重采样、多波段或尺寸不足的输入。"""
    with rasterio.open(pre_image) as pre_ds, rasterio.open(post_image) as post_ds:
        if pre_ds.count != 1 or post_ds.count != 1:
            raise ValueError("Pre/Post SAR inputs must both be single-band GeoTIFFs")
        if (pre_ds.width, pre_ds.height) != (post_ds.width, post_ds.height):
            raise ValueError("Pre/Post image dimensions differ")
        if pre_ds.crs != post_ds.crs:
            raise ValueError("Pre/Post CRS differ; implicit reprojection is not supported")
        if pre_ds.transform != post_ds.transform:
            raise ValueError("Pre/Post transforms differ; inputs must share one pixel grid")
        if pre_ds.width < tile_size or pre_ds.height < tile_size:
            raise ValueError("Input dimensions must both be at least 256 pixels")
        metadata = _source_metadata(pre_ds)
        metadata.update(
            {
                "pre_name": pre_image.name,
                "post_name": post_image.name,
                "pre_dtype": pre_ds.dtypes[0],
                "post_dtype": post_ds.dtypes[0],
                "pre_nodata": _safe_nodata(pre_ds.nodata),
                "post_nodata": _safe_nodata(post_ds.nodata),
                "pre_image": str(pre_image.resolve()),
                "post_image": str(post_image.resolve()),
                "pre_size_bytes": pre_image.stat().st_size,
                "post_size_bytes": post_image.stat().st_size,
                "pre_mtime_ns": pre_image.stat().st_mtime_ns,
                "post_mtime_ns": post_image.stat().st_mtime_ns,
                "pre_units": pre_ds.units[0] or pre_ds.tags().get("units"),
                "post_units": post_ds.units[0] or post_ds.tags().get("units"),
                "pre_pixel_value": pre_ds.tags().get("pixel_value"),
                "post_pixel_value": post_ds.tags().get("pixel_value"),
            }
        )
        metadata.pop("dtype")
        metadata.pop("nodata")
        return metadata



def _guard_output_target(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    protected = {
        Path("/").resolve(),
        Path.home().resolve(),
        Path.cwd().resolve(),
        Path(__file__).resolve().parents[1],
    }
    if resolved in protected:
        raise ValueError(f"Refusing to use a protected directory as output: {resolved}")



def _prepare_output(output_dir: Path, *, overwrite: bool) -> None:
    _guard_output_target(output_dir)
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"Output path exists and is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}; pass --overwrite to replace image tiles"
            )
    # 只清理本程序拥有的影像目录，标签、标签报告及其他用户文件不属于清理范围。
    for relative in (TILE_SUBDIR / "A", TILE_SUBDIR / "B", TILE_SUBDIR / "valid_mask"):
        target = output_dir / relative
        if overwrite:
            if target.is_symlink() or target.is_file():
                target.unlink()
            elif target.is_dir():
                shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)



def _write_rgb_png(gray: np.ndarray, path: Path) -> None:
    Image.fromarray(np.repeat(gray[:, :, None], 3, axis=2), mode="RGB").save(path)



def _write_valid_mask(
    source: rasterio.io.DatasetReader,
    valid_mask: np.ndarray,
    window: Window,
    path: Path,
) -> None:
    profile = source.profile.copy()
    profile.update(
        driver="GTiff",
        height=int(window.height),
        width=int(window.width),
        count=1,
        dtype="uint8",
        transform=rasterio.windows.transform(window, source.transform),
        compress="LZW",
        nodata=0,
    )
    with rasterio.open(path, "w", **profile) as destination:
        destination.write(valid_mask.astype(np.uint8), 1)



def _existing_label_rows(output_dir: Path, source: dict[str, Any]) -> dict[str, dict[str, str]]:
    """仅同一源场景可复用标签清单引用；文件本身始终保留。"""
    manifest = output_dir / "tile_manifest.csv"
    report_path = output_dir / "prepare_report.json"
    if not manifest.is_file() or not report_path.is_file():
        return {}
    old_source = json.loads(report_path.read_text(encoding="utf-8")).get("source")
    if old_source != source:
        print("[NOTICE] Source metadata changed; existing label files are retained but "
              "label references are not reused. Rebuild labels for the new scene.", file=sys.stderr)
        return {}
    with manifest.open(encoding="utf-8", newline="") as stream:
        return {row["tile_id"]: row for row in csv.DictReader(stream)}


def _retain_label_references(
    row: dict[str, Any], previous: dict[str, str], output_dir: Path,
) -> None:
    """按 tile_id 和实际窗口匹配，不把旧标签附到改变后的窗口上。"""
    if any(str(row[key]) != previous.get(key) for key in ("top", "left", "height", "width")):
        return
    for key in ("label_png", "label_tif"):
        value = previous.get(key)
        if value and (output_dir / value).is_file():
            row[key] = value


def _write_manifest(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "tile_index",
        "tile_id",
        "top",
        "left",
        "height",
        "width",
        "valid_ratio",
        "a_png",
        "b_png",
        "valid_mask",
    ]
    fieldnames.extend(key for key in ("label_png", "label_tif") if any(key in row for row in rows))
    with (output_dir / "tile_manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)



def prepare_scene_tiles(options: PrepareOptions) -> dict[str, Any]:
    """验证场景、遍历窗口并生成独立可迁移的推理输入目录。"""
    validate_options(options)
    source = validate_scene_pair(options.pre_image, options.post_image, options.tile_size)
    windows = iter_windows(
        int(source["height"]),
        int(source["width"]),
        options.tile_size,
        options.stride,
    )
    previous_labels = _existing_label_rows(options.output_dir, source) if options.overwrite else {}
    rows: list[dict[str, Any]] = []
    skipped = 0
    with rasterio.open(options.pre_image) as pre_ds, rasterio.open(options.post_image) as post_ds:
        bounds, statistics = fit_scene_stretch(pre_ds, post_ds, options.stretch)
        print(
            f"[STRETCH] {statistics['policy']} low={bounds.low:.12g} high={bounds.high:.12g} "
            f"common_valid={statistics['common_valid_pixels']} "
            f"percentile_error_bound={statistics['percentile_absolute_error_bound']:.6g}",
            file=sys.stderr,
            flush=True,
        )
        print(
            "[NOTICE] New scene-wide shared mapping changes the input distribution of existing "
            "checkpoints; detection accuracy has not been validated.",
            file=sys.stderr,
        )
        if not options.dry_run:
            _prepare_output(options.output_dir, overwrite=options.overwrite)
        for candidate_index, window in enumerate(windows):
            pre, post, valid = read_scene_window(pre_ds, post_ds, window)
            valid_ratio = float(valid.mean())
            if valid_ratio < options.min_valid_ratio:
                skipped += 1
                continue

            top = int(window.row_off)
            left = int(window.col_off)
            tile_id = f"{options.scene_id}_r{top:05d}_c{left:05d}"
            a_relative = TILE_SUBDIR / "A" / f"{tile_id}.png"
            b_relative = TILE_SUBDIR / "B" / f"{tile_id}.png"
            mask_relative = TILE_SUBDIR / "valid_mask" / f"{tile_id}.tif"
            rows.append(
                {
                    "tile_index": candidate_index,
                    "tile_id": tile_id,
                    "top": top,
                    "left": left,
                    "height": int(window.height),
                    "width": int(window.width),
                    "valid_ratio": f"{valid_ratio:.6f}",
                    "a_png": a_relative.as_posix(),
                    "b_png": b_relative.as_posix(),
                    "valid_mask": mask_relative.as_posix(),
                }
            )
            _retain_label_references(rows[-1], previous_labels.get(tile_id, {}), options.output_dir)
            if options.dry_run:
                continue
            _write_rgb_png(
                stretch_to_uint8(pre, valid, bounds),
                options.output_dir / a_relative,
            )
            _write_rgb_png(
                stretch_to_uint8(post, valid, bounds),
                options.output_dir / b_relative,
            )
            _write_valid_mask(pre_ds, valid, window, options.output_dir / mask_relative)

    report = {
        "schema_version": PREPARE_SCHEMA_VERSION,
        "source": source,
        "preprocessing": statistics,
        "params": {
            "scene_id": options.scene_id,
            "tile_size": options.tile_size,
            "stride": options.stride,
            "min_valid_ratio": options.min_valid_ratio,
            "stretch": asdict(options.stretch),
        },
        "tile_counts": {
            "candidate_tiles": len(windows),
            "kept_tiles": len(rows),
            "skipped_tiles": skipped,
        },
        "examples": [row["tile_id"] for row in rows[:5]],
    }
    if not rows:
        raise ValueError("No inference tiles met min_valid_ratio")
    if not options.dry_run:
        _write_manifest(options.output_dir, rows)
        (options.output_dir / "prepare_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return report



def build_parser() -> argparse.ArgumentParser:
    """参数默认值与交付版一致；不读取默认 YAML。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-image", type=Path, required=True, help="灾前单波段 GeoTIFF")
    parser.add_argument("--post-image", type=Path, required=True, help="灾后单波段 GeoTIFF")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--tile-size", type=int, default=256, help="固定为 256")
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--min-valid-ratio", type=float, default=0.01)
    parser.add_argument("--stretch-mode", choices=["percentile", "value"], default="percentile")
    parser.add_argument("--stretch-low", type=float, default=2.0, help="双时相整景联合百分位下限")
    parser.add_argument("--stretch-high", type=float, default=98.0, help="双时相整景联合百分位上限")
    parser.add_argument("--value-min", type=float)
    parser.add_argument("--value-max", type=float)
    parser.add_argument("--dry-run", action="store_true", help="统计并报告，不写文件")
    parser.add_argument("--overwrite", action="store_true", help="重建影像切片及清单/制备报告，保留标签和其他文件")
    return parser


def parse_options(argv: Sequence[str] | None = None) -> PrepareOptions:
    """显式路径相对当前工作目录解析，不引入交付包依赖。"""
    args = build_parser().parse_args(argv)
    return PrepareOptions(
        pre_image=args.pre_image.expanduser().resolve(),
        post_image=args.post_image.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        scene_id=args.scene_id,
        tile_size=args.tile_size,
        stride=args.stride,
        min_valid_ratio=args.min_valid_ratio,
        stretch=StretchConfig(
            mode=args.stretch_mode, low=args.stretch_low, high=args.stretch_high,
            value_min=args.value_min, value_max=args.value_max,
        ),
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )


def main(argv: Sequence[str] | None = None) -> None:
    options = parse_options(argv)
    report = prepare_scene_tiles(options)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    mode = "dry-run" if options.dry_run else "written"
    print(
        f"[DONE] mode={mode} candidate={report['tile_counts']['candidate_tiles']} "
        f"kept={report['tile_counts']['kept_tiles']} skipped={report['tile_counts']['skipped_tiles']}"
    )


if __name__ == "__main__":
    main()
