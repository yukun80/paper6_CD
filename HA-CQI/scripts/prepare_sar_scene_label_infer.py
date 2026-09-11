#!/usr/bin/env python3
"""按现有推理切片清单回切整景 label，并补齐 PNG/TIF 标签文件。"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Sequence

import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window
from rasterio.crs import CRS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.tif_io import (  # noqa: E402
    LABEL_FOREGROUND,
    LABEL_NODATA,
    decode_binary_label,
    read_label_mask,
    read_raster_valid_mask,
)

"""
python HA-CQI/scripts/prepare_sar_scene_label_infer.py \
  --src-root datasets/GF3_Henan \
  --label-image GF3_Zhengzhou_label.tif \
  --tiles-root datasets/GF3_Henan_CD_infer \
  --overwrite

python HA-CQI/scripts/prepare_sar_scene_label_infer.py \
  --src-root datasets/GF3_Zhuozhou \
  --label-image GF3_Zhuozhou_label.tif \
  --tiles-root datasets/GF3_Zhuozhou_CD_infer \
  --overwrite

python HA-CQI/scripts/prepare_sar_scene_label_infer.py \
  --src-root datasets/LT1_Guangxi \
  --label-image LT_Guangxi_Label.tif \
  --tiles-root datasets/LT1_Guangxi_CD_infer \
  --strict

python HA-CQI/scripts/prepare_sar_scene_label_infer.py \
  --src-root datasets/LT1_Guangxi \
  --label-image LT_Guangxi_Label.tif \
  --tiles-root datasets/LT1_Guangxi_CD_infer \
  --strict

"""


@dataclass(frozen=True)
class LabelMeta:
    """记录整景标签 tif 的关键元数据，便于和推理集元信息做一致性校验。"""

    path: Path
    width: int
    height: int
    crs: str | None
    transform: str
    transform_coefficients: tuple[float, ...]
    dtype: str
    nodata: float | int | None


def build_parser(
    description: str = "Prepare SAR scene label tiles for HA-CQI inference",
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description)
    parser.add_argument("--src-root", type=Path, default=Path("datasets/GF3_Henan"))
    parser.add_argument("--label-image", type=str, default="GF3_Zhengzhou_label.tif")
    parser.add_argument("--tiles-root", type=Path, default=Path("datasets/GF3_Henan_CD_infer"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="严格校验 label 为单波段整数 tif，且与 prepare_report.json 中记录的整景元数据一致。",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def ensure_args(args: argparse.Namespace) -> None:
    if not args.src_root.is_dir():
        raise FileNotFoundError(f"Source root not found: {args.src_root}")
    if not args.tiles_root.is_dir():
        raise FileNotFoundError(f"Tiles root not found: {args.tiles_root}")


def load_manifest(manifest_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not fieldnames:
        raise ValueError(f"Manifest has no header: {manifest_path}")
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    # 清单仍是路径来源，但只接受当前 test/ 布局，不重写历史路径。
    path_fields = ("a_png", "b_png", "valid_mask", "a_tif", "b_tif", "label_png", "label_tif")
    for row_index, row in enumerate(rows, start=2):
        for key in path_fields:
            value = row.get(key)
            if not value:
                continue
            relative = PurePosixPath(value)
            if (
                relative.is_absolute() or len(relative.parts) < 2
                or relative.parts[0] != "test" or ".." in relative.parts
            ):
                raise ValueError(
                    f"Manifest row {row_index} {key}={value!r}: only test/ paths are "
                    "supported; legacy tiles/ layout is unsupported. Rebuild image tiles "
                    "with HA-CQI/scripts/prepare_tiles.py."
                )
    return rows, fieldnames


def validate_label(label_path: Path, strict: bool) -> LabelMeta:
    """校验 label tif 基本可读，且满足单波段二值化输入要求。"""
    if not label_path.is_file():
        raise FileNotFoundError(f"Missing label tif: {label_path}")

    with rasterio.open(label_path) as ds:
        if ds.count != 1:
            raise ValueError("Label preprocessing expects single-band tif input")
        if strict and ds.dtypes[0] not in {"uint8", "uint16", "int16", "int32", "uint32"}:
            raise ValueError(f"Strict mode expects integer label tif, got {ds.dtypes[0]}")
        return LabelMeta(
            path=label_path,
            width=ds.width,
            height=ds.height,
            crs=str(ds.crs) if ds.crs else None,
            transform=str(ds.transform),
            transform_coefficients=tuple(ds.transform)[:6],
            dtype=ds.dtypes[0],
            nodata=ds.nodata,
        )


def load_prepare_report(tiles_root: Path) -> dict[str, object] | None:
    report_path = tiles_root / "prepare_report.json"
    if not report_path.is_file():
        return None
    return json.loads(report_path.read_text(encoding="utf-8"))


def validate_against_prepare_report(label: LabelMeta, report: dict[str, object] | None) -> None:
    """兼容新旧报告编码，仍要求标签与场景同 CRS、同像素网格。"""
    if report is None:
        return
    if not isinstance(report, dict):
        raise ValueError("prepare_report.json must contain an object")
    if "source" not in report:
        return
    source = report["source"]
    if not isinstance(source, dict):
        raise ValueError("prepare_report.json source must contain an object")

    expected_width = source.get("width")
    expected_height = source.get("height")
    if expected_width is not None and int(expected_width) != label.width:
        raise ValueError(f"Label width mismatch: {label.width} != {expected_width}")
    if expected_height is not None and int(expected_height) != label.height:
        raise ValueError(f"Label height mismatch: {label.height} != {expected_height}")

    # 多个非空 CRS 表示必须一致，不以优先级掩盖冲突；null 表示该编码未提供。
    reported_crs: list[CRS] = []
    for key in ("crs", "crs_wkt", "crs_epsg"):
        value = source.get(key)
        if value is None:
            continue
        try:
            if key == "crs_epsg":
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ValueError("crs_epsg must be a positive integer")
                parsed = CRS.from_epsg(value)
            else:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{key} must be a nonempty string")
                parsed = CRS.from_wkt(value) if key == "crs_wkt" else CRS.from_user_input(value)
        except (ValueError, TypeError, rasterio.errors.CRSError) as exc:
            raise ValueError(f"Invalid source.{key}: {value!r}") from exc
        if reported_crs and parsed != reported_crs[0]:
            raise ValueError("Conflicting CRS representations in prepare_report.json")
        reported_crs.append(parsed)
    expected_crs = reported_crs[0] if reported_crs else None
    actual_crs = CRS.from_user_input(label.crs) if label.crs is not None else None
    if expected_crs != actual_crs:
        raise ValueError(f"Label CRS mismatch: {actual_crs} != {expected_crs}")

    expected_transform = source.get("transform")
    if isinstance(expected_transform, str):
        # 历史报告保存 Affine 的显示字符串，维持原比较契约。
        matches = expected_transform == label.transform
    elif isinstance(expected_transform, list):
        if len(expected_transform) != 6 or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not np.isfinite(value) for value in expected_transform
        ):
            raise ValueError("source.transform must contain six finite numeric coefficients")
        matches = tuple(expected_transform) == label.transform_coefficients
    else:
        raise ValueError("source.transform must be a legacy string or six numeric coefficients")
    if not matches:
        raise ValueError("Label geotransform mismatch with prepare_report.json")


def ensure_windows_in_bounds(rows: list[dict[str, str]], label: LabelMeta) -> None:
    """确保 manifest 中记录的所有窗口都落在 label 整景范围内。"""
    for row in rows:
        top = int(row["top"])
        left = int(row["left"])
        height = int(row["height"])
        width = int(row["width"])
        if top < 0 or left < 0:
            raise ValueError(f"Negative window offset in manifest: {row['tile_id']}")
        if top + height > label.height or left + width > label.width:
            raise ValueError(f"Window out of label bounds: {row['tile_id']}")


def clean_output_dirs(tiles_root: Path, overwrite: bool, dry_run: bool) -> None:
    for rel_dir in ("test/label", "test/label_tif"):
        out_dir = tiles_root / rel_dir
        if not out_dir.exists():
            continue
        if not overwrite:
            if any(out_dir.iterdir()):
                raise FileExistsError(f"Output directory already exists and is not empty: {out_dir}")
            continue
        if not dry_run:
            shutil.rmtree(out_dir)


def prepare_dirs(tiles_root: Path, dry_run: bool) -> None:
    if dry_run:
        return
    for rel_dir in ("test/label", "test/label_tif"):
        (tiles_root / rel_dir).mkdir(parents=True, exist_ok=True)


def write_png_label(arr: np.ndarray, out_path: Path, dry_run: bool) -> None:
    if dry_run:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="L").save(out_path)


def write_tif_from_window(
    src_ds: rasterio.io.DatasetReader,
    arr: np.ndarray,
    out_path: Path,
    window: Window,
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
            "dtype": "uint8",
            "transform": rasterio.windows.transform(window, src_ds.transform),
            "compress": "LZW",
            "nodata": LABEL_NODATA,
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as out_ds:
        out_ds.write(arr, 1)


def write_outputs(
    tiles_root: Path,
    label_path: Path,
    rows: list[dict[str, str]],
    dry_run: bool,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """按 manifest 回切标签；仅接受 test/ 路径，不重写路径。"""
    updated_rows: list[dict[str, str]] = []
    stats = {
        "valid_pixels": 0, "foreground_pixels": 0, "nodata_pixels": 0,
        "source_gt_invalid_pixels": 0, "sar_invalid_pixels": 0,
    }

    with rasterio.open(label_path) as ds_label:
        for row in rows:
            top = int(row["top"])
            left = int(row["left"])
            height = int(row["height"])
            width = int(row["width"])
            tile_id = row["tile_id"]

            window = Window(col_off=left, row_off=top, width=width, height=height)
            arr = ds_label.read(1, window=window)
            label_binary, source_valid = decode_binary_label(
                arr, ds_label.nodata, read_label_mask(ds_label, window),
            )
            sar_valid = np.ones(arr.shape, dtype=bool)
            # 使用清单已有的 SAR 覆盖信息，不把 PNG 的黑色猜作 NoData。
            for key in ("valid_mask", "a_tif", "b_tif"):
                if row.get(key):
                    with rasterio.open(tiles_root / row[key]) as source:
                        source_array = source.read(1)
                        values = source_array[:height, :width]
                        source_mask = read_raster_valid_mask(source, source_array)[:height, :width]
                    if values.shape != arr.shape:
                        raise ValueError(f"Label/SAR mask shape mismatch for tile {tile_id}: {key}")
                    if key == "valid_mask":
                        source_mask &= values > 0
                    sar_valid &= source_mask
            label_binary[~sar_valid] = 0
            label_png = label_binary * LABEL_FOREGROUND
            label_tif = label_png
            stats["valid_pixels"] += int(arr.size)
            stats["foreground_pixels"] += int(np.count_nonzero(label_binary))
            stats["source_gt_invalid_pixels"] += int(np.count_nonzero(~source_valid))
            stats["sar_invalid_pixels"] += int(np.count_nonzero(~sar_valid))

            label_png_rel = Path("test/label") / f"{tile_id}.png"
            label_tif_rel = Path("test/label_tif") / f"{tile_id}.tif"
            write_png_label(label_png, tiles_root / label_png_rel, dry_run)
            write_tif_from_window(ds_label, label_tif, tiles_root / label_tif_rel, window, dry_run)

            new_row = dict(row)
            new_row["label_png"] = str(label_png_rel)
            new_row["label_tif"] = str(label_tif_rel)
            updated_rows.append(new_row)

    return updated_rows, stats


def write_manifest(
    manifest_path: Path,
    rows: list[dict[str, str]],
    fieldnames: list[str],
    dry_run: bool,
) -> None:
    if dry_run:
        return

    final_fields = list(fieldnames)
    for extra in ("label_png", "label_tif"):
        if extra not in final_fields:
            final_fields.append(extra)

    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=final_fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    tiles_root: Path,
    label: LabelMeta,
    row_count: int,
    pixel_stats: dict[str, int],
    dry_run: bool,
) -> None:
    payload = {
        "label_source": {
            "path": str(label.path.resolve()),
            "width": label.width,
            "height": label.height,
            "crs": label.crs,
            "transform": label.transform,
            "dtype": label.dtype,
            "nodata": label.nodata,
        },
        "outputs": {
            "label_png_dir": "test/label",
            "label_tif_dir": "test/label_tif",
        },
        "tile_count": row_count,
        "label_encoding": {"background": 0, "foreground": LABEL_FOREGROUND, "nodata": LABEL_NODATA},
        "label_policy": "unknown_gt_and_invalid_sar_as_background",
        "tile_pixel_stats": pixel_stats,
    }
    if dry_run:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    (tiles_root / "label_prepare_report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.src_root = Path(args.src_root)
    args.tiles_root = Path(args.tiles_root)
    ensure_args(args)

    manifest_path = args.tiles_root / "tile_manifest.csv"
    label_path = args.src_root / args.label_image

    rows, fieldnames = load_manifest(manifest_path)
    label = validate_label(label_path, args.strict)
    validate_against_prepare_report(label, load_prepare_report(args.tiles_root))
    ensure_windows_in_bounds(rows, label)

    clean_output_dirs(args.tiles_root, args.overwrite, args.dry_run)
    prepare_dirs(args.tiles_root, args.dry_run)
    updated_rows, pixel_stats = write_outputs(args.tiles_root, label_path, rows, args.dry_run)
    write_manifest(manifest_path, updated_rows, fieldnames, args.dry_run)
    write_report(
        args.tiles_root,
        label,
        row_count=len(updated_rows),
        pixel_stats=pixel_stats,
        dry_run=args.dry_run,
    )

    print(f"[DONE] label_tiles={len(updated_rows)} manifest={manifest_path}")


if __name__ == "__main__":
    main()
