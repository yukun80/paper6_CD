#!/usr/bin/env python3
"""构建 HA-CQI 训练所需的 S1GFloods 与 VarFloods(PRO) 融合数据集。"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window

"""
python HA-CQI/scripts/prepare_fused_sar_cd_dataset.py \
  --s1gfloods-root datasets/S1GFloods \
  --varfloods-root datasets/VarFloods \
  --out-root datasets/S1GFloods_CD_DINO \
  --tile-size 256 \
  --stride 128 \
  --train-ratio 0.8 \
  --seed 42 \
  --overwrite
"""

VALID_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class SampleRecord:
    """描述一个待输出样本，统一承载 S1GFloods 原样本与 VarFloods 切片元数据。"""

    sample_id: str
    source: str
    region: str
    a_path: Path
    b_path: Path
    label_path: Path
    row_off: int | None = None
    col_off: int | None = None
    width: int | None = None
    height: int | None = None
    valid_ratio: float | None = None

    @property
    def is_tiled(self) -> bool:
        return self.row_off is not None and self.col_off is not None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Prepare fused S1GFloods + VarFloods CD dataset")
    parser.add_argument("--s1gfloods-root", type=Path, default=Path("datasets/S1GFloods"))
    parser.add_argument("--varfloods-root", type=Path, default=Path("datasets/VarFloods"))
    parser.add_argument("--out-root", type=Path, default=Path("datasets/S1GFloods_CD_DINO"))
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stretch-low", type=float, default=2.0)
    parser.add_argument("--stretch-high", type=float, default=98.0)
    parser.add_argument(
        "--min-valid-ratio",
        type=float,
        default=0.0,
        help="仅保留有效像素比例 >= 该阈值的 VarFloods 切片；0 表示只剔除全无效切片。",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def ensure_args(args: argparse.Namespace) -> None:
    if not args.s1gfloods_root.is_dir():
        raise FileNotFoundError(f"Missing S1GFloods root: {args.s1gfloods_root}")
    if not args.varfloods_root.is_dir():
        raise FileNotFoundError(f"Missing VarFloods root: {args.varfloods_root}")
    if not 0 < args.tile_size:
        raise ValueError("--tile-size must be positive")
    if not 0 < args.stride:
        raise ValueError("--stride must be positive")
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be within (0, 1)")
    if not 0.0 <= args.min_valid_ratio <= 1.0:
        raise ValueError("--min-valid-ratio must be within [0, 1]")
    if not 0.0 <= args.stretch_low < args.stretch_high <= 100.0:
        raise ValueError("--stretch-low/--stretch-high must satisfy 0 <= low < high <= 100")
    if args.out_root.exists() and not args.overwrite and not args.dry_run:
        raise FileExistsError(f"Output root exists, rerun with --overwrite: {args.out_root}")


def scan_files(folder: Path) -> dict[str, Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing directory: {folder}")
    files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_IMAGE_SUFFIXES)
    return {p.name: p for p in files}


def build_s1gfloods_records(src_root: Path, strict: bool) -> list[SampleRecord]:
    """读取扁平 S1GFloods 目录，并给样本增加统一前缀避免与 VarFloods 重名。"""
    a_map = scan_files(src_root / "A")
    b_map = scan_files(src_root / "B")
    label_dir = src_root / "Label"
    if not label_dir.is_dir():
        label_dir = src_root / "label"
    label_map = scan_files(label_dir)

    if sorted(a_map) != sorted(b_map) or sorted(a_map) != sorted(label_map):
        raise ValueError("S1GFloods A/B/Label file names are not aligned")

    records: list[SampleRecord] = []
    for name in sorted(a_map):
        if strict and Path(name).suffix.lower() != ".png":
            raise ValueError(f"S1GFloods strict mode expects PNG only, got: {name}")
        stem = Path(name).stem
        sample_id = f"s1gfloods_{stem}"
        records.append(
            SampleRecord(
                sample_id=sample_id,
                source="s1gfloods",
                region="S1GFloods",
                a_path=a_map[name],
                b_path=b_map[name],
                label_path=label_map[name],
            )
        )
    return records


def build_positions(length: int, tile_size: int, stride: int) -> list[int]:
    """生成覆盖右边界/下边界的窗口起始坐标。"""
    if length <= tile_size:
        return [0]
    positions = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return sorted(set(positions))


def iter_windows(height: int, width: int, tile_size: int, stride: int) -> list[Window]:
    windows: list[Window] = []
    for top in build_positions(height, tile_size, stride):
        for left in build_positions(width, tile_size, stride):
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


def stretch_to_uint8(arr: np.ndarray, valid_mask: np.ndarray, low: float, high: float) -> np.ndarray:
    """按有效像素百分位做稳定拉伸，保持与现有 SAR 推理脚本一致。"""
    out = np.zeros(arr.shape, dtype=np.uint8)
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

    scaled = np.zeros(arr.shape, dtype=np.float32)
    scaled[valid_mask] = np.clip((arr[valid_mask].astype(np.float32, copy=False) - lo) / (hi - lo), 0.0, 1.0)
    out[valid_mask] = np.rint(scaled[valid_mask] * 255.0).astype(np.uint8)
    return out


def label_to_uint8(arr: np.ndarray) -> np.ndarray:
    """统一把标签转成 0/255，便于训练阶段沿用现有二值读取逻辑。"""
    return np.where(arr > 0, 255, 0).astype(np.uint8)


def has_foreground_pixels(arr: np.ndarray) -> bool:
    """判断切片标签中是否存在前景像素，用于剔除全背景样本。"""
    return bool(np.any(arr > 0))


def find_single_tif(folder: Path) -> Path:
    tif_files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".tif", ".tiff"})
    if len(tif_files) != 1:
        raise ValueError(f"Expected exactly 1 tif under {folder}, got {len(tif_files)}")
    return tif_files[0]


def region_transform_summary(ds: rasterio.io.DatasetReader) -> dict[str, object]:
    return {
        "crs": str(ds.crs) if ds.crs else None,
        "transform": str(ds.transform),
        "resolution": [float(ds.transform.a), float(ds.transform.e)],
        "bounds": [float(v) for v in ds.bounds],
        "dtype": ds.dtypes[0],
        "nodata": None if ds.nodata is None else float(ds.nodata),
        "shape": [int(ds.height), int(ds.width)],
    }


def build_varfloods_records(args: argparse.Namespace) -> tuple[list[SampleRecord], list[dict[str, object]]]:
    """扫描 VarFloods PRO 整景并生成切片元数据，不在这个阶段落盘。"""
    records: list[SampleRecord] = []
    region_reports: list[dict[str, object]] = []

    region_dirs = sorted(p for p in args.varfloods_root.iterdir() if p.is_dir())
    for region_dir in region_dirs:
        pro_root = region_dir / "PRO"
        a_path = find_single_tif(pro_root / "A")
        b_path = find_single_tif(pro_root / "B")
        label_path = find_single_tif(pro_root / "label")

        with rasterio.open(a_path) as ds_a, rasterio.open(b_path) as ds_b, rasterio.open(label_path) as ds_l:
            if ds_a.count != 1 or ds_b.count != 1 or ds_l.count != 1:
                raise ValueError(f"{region_dir.name}: PRO A/B/label must all be single-band tif")
            if (ds_a.height, ds_a.width) != (ds_b.height, ds_b.width) or (ds_a.height, ds_a.width) != (
                ds_l.height,
                ds_l.width,
            ):
                raise ValueError(f"{region_dir.name}: PRO A/B/label shape mismatch")
            if ds_a.crs != ds_b.crs or ds_a.crs != ds_l.crs:
                raise ValueError(f"{region_dir.name}: PRO A/B/label CRS mismatch")
            if args.strict and ds_l.dtypes[0] not in {"uint8", "uint16"}:
                raise ValueError(f"{region_dir.name}: strict mode expects uint8/uint16 label tif, got {ds_l.dtypes[0]}")

            windows = iter_windows(ds_a.height, ds_a.width, args.tile_size, args.stride)
            kept = 0
            skipped_invalid = 0
            skipped_background_only = 0
            for window in windows:
                top = int(window.row_off)
                left = int(window.col_off)
                arr_a = ds_a.read(1, window=window).astype(np.float32, copy=False)
                arr_b = ds_b.read(1, window=window).astype(np.float32, copy=False)
                valid_mask = build_valid_mask(arr_a, ds_a.nodata, arr_b, ds_b.nodata)
                valid_ratio = float(valid_mask.mean())
                if valid_ratio <= 0.0 or valid_ratio < args.min_valid_ratio:
                    skipped_invalid += 1
                    continue

                arr_l = ds_l.read(1, window=window)
                if not has_foreground_pixels(arr_l):
                    skipped_background_only += 1
                    continue

                kept += 1
                sample_id = f"varfloods_{region_dir.name.lower()}_pro_r{top:05d}_c{left:05d}"
                records.append(
                    SampleRecord(
                        sample_id=sample_id,
                        source="varfloods_pro",
                        region=region_dir.name,
                        a_path=a_path,
                        b_path=b_path,
                        label_path=label_path,
                        row_off=top,
                        col_off=left,
                        width=int(window.width),
                        height=int(window.height),
                        valid_ratio=valid_ratio,
                    )
                )

            region_reports.append(
                {
                    "region": region_dir.name,
                    "candidate_tiles": len(windows),
                    "kept_tiles": kept,
                    "skipped_tiles": skipped_invalid + skipped_background_only,
                    "skipped_invalid_tiles": skipped_invalid,
                    "skipped_background_only_tiles": skipped_background_only,
                    "a_meta": region_transform_summary(ds_a),
                    "b_meta": region_transform_summary(ds_b),
                    "label_meta": region_transform_summary(ds_l),
                    "a_b_transform_equal": ds_a.transform == ds_b.transform,
                    "a_label_transform_equal": ds_a.transform == ds_l.transform,
                }
            )

    return records, region_reports


def assign_splits(records: list[SampleRecord], train_ratio: float, seed: int) -> dict[str, list[SampleRecord]]:
    rng = random.Random(seed)
    shuffled = records.copy()
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = n_total - n_train
    if min(n_train, n_val) <= 0:
        raise ValueError("Split produces an empty subset; adjust --train-ratio")
    return {"train": shuffled[:n_train], "val": shuffled[n_train:]}


def clean_output_root(out_root: Path, overwrite: bool, dry_run: bool) -> None:
    if not out_root.exists():
        return
    if not overwrite or dry_run:
        return

    # 某些文件系统上 rmtree 可能偶发报 "Directory not empty"，这里做一次稳妥兜底。
    for attempt in range(3):
        try:
            shutil.rmtree(out_root)
            return
        except FileNotFoundError:
            return
        except OSError:
            if not out_root.exists():
                return
            time.sleep(0.2 * (attempt + 1))

    for root, dirnames, filenames in os.walk(out_root, topdown=False):
        root_path = Path(root)
        for filename in filenames:
            path = root_path / filename
            if path.exists() or path.is_symlink():
                path.unlink()
        for dirname in dirnames:
            path = root_path / dirname
            if path.exists():
                path.rmdir()
    if out_root.exists():
        out_root.rmdir()


def prepare_dirs(out_root: Path, dry_run: bool) -> None:
    for split in ("train", "val"):
        for sub in ("A", "B", "label"):
            if not dry_run:
                (out_root / split / sub).mkdir(parents=True, exist_ok=True)
                (out_root / f"{split}_tif" / sub).mkdir(parents=True, exist_ok=True)


def write_png_rgb(arr: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.repeat(arr[:, :, None], 3, axis=2)
    Image.fromarray(rgb, mode="RGB").save(out_path)


def write_png_label(arr: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="L").save(out_path)


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def write_tif_from_window(
    src_ds: rasterio.io.DatasetReader,
    arr: np.ndarray,
    out_path: Path,
    window: Window,
    dtype: str,
    nodata: float | int | None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
    with rasterio.open(out_path, "w", **profile) as out_ds:
        out_ds.write(arr, 1)


def manifest_row(record: SampleRecord, split: str, out_root: Path, write_tif: bool) -> dict[str, object]:
    base_png = Path(split)
    base_tif = Path(f"{split}_tif")
    row = {
        "sample_id": record.sample_id,
        "source": record.source,
        "region": record.region,
        "split": split,
        "a_png": str(base_png / "A" / f"{record.sample_id}.png"),
        "b_png": str(base_png / "B" / f"{record.sample_id}.png"),
        "label_png": str(base_png / "label" / f"{record.sample_id}.png"),
        "a_tif": str(base_tif / "A" / f"{record.sample_id}.tif") if write_tif else "",
        "b_tif": str(base_tif / "B" / f"{record.sample_id}.tif") if write_tif else "",
        "label_tif": str(base_tif / "label" / f"{record.sample_id}.tif") if write_tif else "",
        "row_off": "" if record.row_off is None else record.row_off,
        "col_off": "" if record.col_off is None else record.col_off,
        "height": "" if record.height is None else record.height,
        "width": "" if record.width is None else record.width,
        "valid_ratio": "" if record.valid_ratio is None else f"{record.valid_ratio:.6f}",
    }
    return row


def write_s1_record(record: SampleRecord, out_root: Path, split: str) -> None:
    """S1GFloods 已经是 PNG，对应 split 下直接重命名复制。"""
    copy_file(record.a_path, out_root / split / "A" / f"{record.sample_id}.png")
    copy_file(record.b_path, out_root / split / "B" / f"{record.sample_id}.png")
    copy_file(record.label_path, out_root / split / "label" / f"{record.sample_id}.png")


def write_var_records(records: list[SampleRecord], out_root: Path, split: str, args: argparse.Namespace) -> None:
    """按区域分组写 VarFloods 切片，减少重复打开整景 tif 的开销。"""
    grouped: dict[tuple[Path, Path, Path], list[SampleRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.a_path, record.b_path, record.label_path)].append(record)

    for (a_path, b_path, label_path), group in grouped.items():
        with rasterio.open(a_path) as ds_a, rasterio.open(b_path) as ds_b, rasterio.open(label_path) as ds_l:
            for record in group:
                window = Window(col_off=record.col_off, row_off=record.row_off, width=record.width, height=record.height)
                arr_a = ds_a.read(1, window=window).astype(np.float32, copy=False)
                arr_b = ds_b.read(1, window=window).astype(np.float32, copy=False)
                arr_l = ds_l.read(1, window=window)

                valid_mask = build_valid_mask(arr_a, ds_a.nodata, arr_b, ds_b.nodata)
                a_png = stretch_to_uint8(arr_a, valid_mask, args.stretch_low, args.stretch_high)
                b_png = stretch_to_uint8(arr_b, valid_mask, args.stretch_low, args.stretch_high)
                label_png = label_to_uint8(arr_l)

                write_png_rgb(a_png, out_root / split / "A" / f"{record.sample_id}.png")
                write_png_rgb(b_png, out_root / split / "B" / f"{record.sample_id}.png")
                write_png_label(label_png, out_root / split / "label" / f"{record.sample_id}.png")

                write_tif_from_window(
                    ds_a,
                    arr_a.astype(ds_a.dtypes[0], copy=False),
                    out_root / f"{split}_tif" / "A" / f"{record.sample_id}.tif",
                    window,
                    ds_a.dtypes[0],
                    ds_a.nodata,
                )
                write_tif_from_window(
                    ds_b,
                    arr_b.astype(ds_b.dtypes[0], copy=False),
                    out_root / f"{split}_tif" / "B" / f"{record.sample_id}.tif",
                    window,
                    ds_b.dtypes[0],
                    ds_b.nodata,
                )
                write_tif_from_window(
                    ds_l,
                    label_png,
                    out_root / f"{split}_tif" / "label" / f"{record.sample_id}.tif",
                    window,
                    "uint8",
                    0,
                )


def write_split_outputs(
    out_root: Path,
    split: str,
    records: list[SampleRecord],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    manifest_rows: list[dict[str, object]] = []

    s1_records = [record for record in records if record.source == "s1gfloods"]
    var_records = [record for record in records if record.source == "varfloods_pro"]

    for record in s1_records:
        if not args.dry_run:
            write_s1_record(record, out_root, split)
        manifest_rows.append(manifest_row(record, split, out_root, write_tif=False))

    if not args.dry_run:
        write_var_records(var_records, out_root, split, args)
    manifest_rows.extend(manifest_row(record, split, out_root, write_tif=True) for record in var_records)
    return manifest_rows


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "sample_id",
        "source",
        "region",
        "split",
        "a_png",
        "b_png",
        "label_png",
        "a_tif",
        "b_tif",
        "label_tif",
        "row_off",
        "col_off",
        "height",
        "width",
        "valid_ratio",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    args: argparse.Namespace,
    splits: dict[str, list[SampleRecord]],
    split_rows: dict[str, list[dict[str, object]]],
    region_reports: list[dict[str, object]],
) -> None:
    payload = {
        "source_roots": {
            "s1gfloods_root": str(args.s1gfloods_root),
            "varfloods_root": str(args.varfloods_root),
            "out_root": str(args.out_root),
        },
        "params": {
            "tile_size": args.tile_size,
            "stride": args.stride,
            "train_ratio": args.train_ratio,
            "val_ratio": 1.0 - args.train_ratio,
            "seed": args.seed,
            "stretch_low": args.stretch_low,
            "stretch_high": args.stretch_high,
            "min_valid_ratio": args.min_valid_ratio,
            "strict": args.strict,
            "dry_run": args.dry_run,
        },
        "counts": {split: len(items) for split, items in splits.items()},
        "counts_by_source": {
            split: dict(Counter(record.source for record in items)) for split, items in splits.items()
        },
        "examples": {
            split: [record.sample_id for record in items[:5]] for split, items in splits.items()
        },
        "varfloods_regions": region_reports,
    }

    if args.dry_run:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    (args.out_root / "split_report.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    all_rows = split_rows["train"] + split_rows["val"]
    write_manifest(args.out_root / "manifest_all.csv", all_rows)
    write_manifest(args.out_root / "manifest_train.csv", split_rows["train"])
    write_manifest(args.out_root / "manifest_val.csv", split_rows["val"])


def main() -> None:
    args = parse_args()
    ensure_args(args)

    s1_records = build_s1gfloods_records(args.s1gfloods_root, args.strict)
    var_records, region_reports = build_varfloods_records(args)
    all_records = s1_records + var_records
    splits = assign_splits(all_records, args.train_ratio, args.seed)

    clean_output_root(args.out_root, args.overwrite, args.dry_run)
    prepare_dirs(args.out_root, args.dry_run)

    split_rows: dict[str, list[dict[str, object]]] = {}
    for split, records in splits.items():
        split_rows[split] = write_split_outputs(args.out_root, split, records, args)

    write_report(args, splits, split_rows, region_reports)
    counts = {split: len(items) for split, items in splits.items()}
    print(
        "[DONE] total={total} train={train} val={val} s1gfloods={s1} varfloods={var}".format(
            total=len(all_records),
            train=counts["train"],
            val=counts["val"],
            s1=len(s1_records),
            var=len(var_records),
        )
    )


if __name__ == "__main__":
    main()
