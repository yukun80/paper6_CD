#!/usr/bin/env python3
"""将 GF3 河南 depth 整景 tif 切成彩色 PNG 可视化瓦片。"""

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
class ColorBin:
    """记录 depth 可视化色带的闭区间和 RGB 颜色。"""

    lower: float
    upper: float
    hex_color: str
    rgb: tuple[int, int, int]


COLOR_BINS: tuple[ColorBin, ...] = (
    ColorBin(0.001, 0.25, "#FFC800", (255, 200, 0)),
    ColorBin(0.251, 0.5, "#FF7D03", (255, 125, 3)),
    ColorBin(0.501, 1.0, "#FF173A", (255, 23, 58)),
    ColorBin(1.001, 2.0, "#FF006F", (255, 0, 111)),
    ColorBin(2.001, 4.0, "#CC00B8", (204, 0, 184)),
    ColorBin(4.001, 200.0, "#0000FF", (0, 0, 255)),
)
BLACK_RGB = (0, 0, 0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Prepare GF3 Henan depth RGB visualization tiles")
    parser.add_argument(
        "--depth-tif",
        type=Path,
        default=Path("datasets/GF3_Henan/depth/FwDET_v2_Zhengzhou_adaptive_nodata.tif"),
        help="输入 depth GeoTIFF。",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("datasets/GF3_Henan_CD_infer/depth_vis"),
        help="输出根目录，仅写入 PNG、manifest 和 report。",
    )
    parser.add_argument(
        "--tiles-root",
        type=Path,
        default=Path("datasets/GF3_Henan_CD_infer"),
        help="已有 GF3 河南推理切片根目录，用于复用 tile_manifest.csv 的切片名和窗口。",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="可选：显式指定已有推理切片 manifest；默认使用 <tiles-root>/tile_manifest.csv。",
    )
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有输出目录。")
    parser.add_argument("--dry-run", action="store_true", help="只打印报告，不写文件。")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def ensure_args(args: argparse.Namespace) -> None:
    if not args.depth_tif.is_file():
        raise FileNotFoundError(f"Missing depth tif: {args.depth_tif}")
    if not args.tiles_root.is_dir():
        raise FileNotFoundError(f"Missing tiles root: {args.tiles_root}")
    if not args.manifest.is_file():
        raise FileNotFoundError(f"Missing tile manifest: {args.manifest}")


def clean_output_root(out_root: Path, overwrite: bool, dry_run: bool) -> None:
    """按需清理输出目录，避免新旧切片混在一起。"""
    if not out_root.exists() or dry_run:
        return
    if not overwrite:
        raise FileExistsError(f"Output root exists, pass --overwrite to replace it: {out_root}")
    shutil.rmtree(out_root)


def build_valid_mask(arr: np.ndarray, nodata: float | int | None) -> np.ndarray:
    """过滤 nodata 和非有限值；其余是否上色由色带闭区间决定。"""
    valid = np.isfinite(arr)
    if nodata is not None:
        valid &= np.not_equal(arr, nodata)
    return valid


def colorize_depth(arr: np.ndarray, nodata: float | int | None) -> np.ndarray:
    """按固定 depth 色带生成 RGB，可疑值和未命中值保持黑色。"""
    rgb = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
    valid = build_valid_mask(arr, nodata)

    for color_bin in COLOR_BINS:
        mask = valid & (arr >= color_bin.lower) & (arr <= color_bin.upper)
        rgb[mask] = color_bin.rgb

    return rgb


def write_png(rgb: np.ndarray, out_path: Path, dry_run: bool) -> None:
    if dry_run:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(out_path)


def write_manifest(out_root: Path, rows: list[dict[str, object]], dry_run: bool) -> None:
    if dry_run:
        return
    fieldnames = ["tile_id", "top", "left", "height", "width", "png_path", "tile_index"]
    with (out_root / "depth_tile_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_source_manifest(manifest_path: Path) -> list[dict[str, object]]:
    """读取已有 SAR 推理切片 manifest，保证 depth 文件名和窗口完全对齐。"""
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    required_fields = {"tile_id", "top", "left", "height", "width", "tile_index"}
    missing = required_fields.difference(rows[0].keys() if rows else set())
    if missing:
        raise ValueError(f"Manifest missing required fields: {sorted(missing)}")
    return rows


def write_report(
    args: argparse.Namespace,
    *,
    width: int,
    height: int,
    crs: str | None,
    transform: str,
    dtype: str,
    nodata: float | int | None,
    tile_count: int,
    rows: list[dict[str, object]],
) -> None:
    payload = {
        "source": {
            "depth_tif": str(args.depth_tif.resolve()),
            "width": width,
            "height": height,
            "crs": crs,
            "transform": transform,
            "dtype": dtype,
            "nodata": nodata,
        },
        "params": {
            "out_root": str(args.out_root),
            "tiles_root": str(args.tiles_root),
            "manifest": str(args.manifest),
            "dry_run": args.dry_run,
        },
        "color_bins": [
            {"lower": item.lower, "upper": item.upper, "hex": item.hex_color}
            for item in COLOR_BINS
        ],
        "fallback_color": {"hex": "#000000", "rgb": BLACK_RGB},
        "tile_counts": {
            "candidate_tiles": tile_count,
            "written_tiles": len(rows),
        },
        "examples": [row["tile_id"] for row in rows[:5]],
    }
    if args.dry_run:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    (args.out_root / "depth_prepare_report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_outputs(args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, object]]:
    source_rows = read_source_manifest(args.manifest)
    rows: list[dict[str, object]] = []
    with rasterio.open(args.depth_tif) as ds:
        if ds.count != 1:
            raise ValueError(f"Depth visualization expects a single-band tif, got {ds.count}")

        for source_row in source_rows:
            tile_id = source_row["tile_id"]
            top = int(source_row["top"])
            left = int(source_row["left"])
            height = int(source_row["height"])
            width = int(source_row["width"])
            tile_index = int(source_row["tile_index"])
            window = Window(col_off=left, row_off=top, width=width, height=height)
            if top < 0 or left < 0 or top + height > ds.height or left + width > ds.width:
                raise ValueError(
                    f"Manifest window out of depth bounds: {tile_id} "
                    f"top={top} left={left} height={height} width={width} "
                    f"depth_shape=({ds.height}, {ds.width})"
                )
            arr = ds.read(1, window=window).astype(np.float32, copy=False)
            rgb = colorize_depth(arr, ds.nodata)
            png_rel = Path("test/depth") / f"{tile_id}.png"

            write_png(rgb, args.out_root / png_rel, args.dry_run)
            rows.append(
                {
                    "tile_id": tile_id,
                    "top": top,
                    "left": left,
                    "height": height,
                    "width": width,
                    "png_path": str(png_rel),
                    "tile_index": tile_index,
                }
            )

        meta = {
            "width": ds.width,
            "height": ds.height,
            "crs": str(ds.crs) if ds.crs else None,
            "transform": str(ds.transform),
            "dtype": ds.dtypes[0],
            "nodata": ds.nodata,
            "tile_count": len(source_rows),
        }
    return rows, meta


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.depth_tif = Path(args.depth_tif)
    args.out_root = Path(args.out_root)
    args.tiles_root = Path(args.tiles_root)
    args.manifest = Path(args.manifest) if args.manifest is not None else args.tiles_root / "tile_manifest.csv"
    ensure_args(args)

    clean_output_root(args.out_root, args.overwrite, args.dry_run)
    rows, meta = write_outputs(args)
    write_manifest(args.out_root, rows, args.dry_run)
    write_report(args, rows=rows, **meta)

    print(f"[DONE] candidate={meta['tile_count']} written={len(rows)} out_root={args.out_root}")


if __name__ == "__main__":
    main()
