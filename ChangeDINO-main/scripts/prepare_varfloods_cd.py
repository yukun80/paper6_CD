#!/usr/bin/env python3
"""将 VarFloods 的 PRO tif 切片并整理为 ChangeDINO 可直接读取的目录。"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

"""
python ChangeDINO-main/scripts/prepare_varfloods_cd.py \
  --src-root datasets/VarFloods \
  --out-root datasets/VarFloods_CD \
  --tile-size 256 \
  --stride 256 \
  --seed 42 \
  --overwrite
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Prepare VarFloods in ChangeDINO CD layout")
    parser.add_argument("--src-root", type=Path, default=Path("datasets/VarFloods"))
    parser.add_argument("--out-root", type=Path, default=Path("datasets/VarFloods_CD"))
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def ensure_args(args: argparse.Namespace) -> None:
    if not args.src_root.is_dir():
        raise FileNotFoundError(f"Source root not found: {args.src_root}")
    if args.tile_size <= 0 or args.stride <= 0:
        raise ValueError("tile-size and stride must be positive")
    if not 0 < args.train_ratio < 1:
        raise ValueError("--train-ratio must be within (0, 1)")
    if not 0 <= args.val_ratio < 1:
        raise ValueError("--val-ratio must be within [0, 1)")
    if args.train_ratio + args.val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1")


def scan_tif_map(folder: Path) -> dict[str, Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing directory: {folder}")
    files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".tif", ".tiff"})
    return {p.name: p for p in files}


def build_scenes(src_root: Path, strict: bool) -> list[dict[str, str | Path]]:
    scenes: list[dict[str, str | Path]] = []
    for region_dir in sorted(p for p in src_root.iterdir() if p.is_dir() and p.name != "tiles"):
        pro_root = region_dir / "PRO"
        if not pro_root.is_dir():
            continue
        a_map = scan_tif_map(pro_root / "A")
        b_map = scan_tif_map(pro_root / "B")
        label_map = scan_tif_map(pro_root / "label")
        names = sorted(a_map)
        if names != sorted(b_map) or names != sorted(label_map):
            raise ValueError(f"A/B/label file names are not aligned under: {pro_root}")
        for name in names:
            if strict and Path(name).suffix.lower() not in {".tif", ".tiff"}:
                raise ValueError(f"Expected tif only, got: {name}")
            stem = Path(name).stem
            scenes.append(
                {
                    "region": region_dir.name,
                    "scene_id": f"{region_dir.name}/{stem}",
                    "stem": stem,
                    "A": a_map[name],
                    "B": b_map[name],
                    "label": label_map[name],
                }
            )
    if not scenes:
        raise ValueError(f"No PRO scenes found under {src_root}")
    return scenes


def iter_windows(height: int, width: int, tile_size: int, stride: int):
    for top in range(0, height - tile_size + 1, stride):
        for left in range(0, width - tile_size + 1, stride):
            yield Window(col_off=left, row_off=top, width=tile_size, height=tile_size)


def validate_scene_shapes(scene: dict[str, str | Path]) -> tuple[int, int]:
    with rasterio.open(scene["A"]) as ds_a, rasterio.open(scene["B"]) as ds_b, rasterio.open(scene["label"]) as ds_l:
        shape_a = (ds_a.height, ds_a.width)
        shape_b = (ds_b.height, ds_b.width)
        shape_l = (ds_l.height, ds_l.width)
        if shape_a != shape_b or shape_a != shape_l:
            raise ValueError(f"Shape mismatch in scene {scene['scene_id']}: {shape_a}, {shape_b}, {shape_l}")
        return shape_a


def build_tile_items(args: argparse.Namespace, scenes: list[dict[str, str | Path]]) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for scene in scenes:
        height, width = validate_scene_shapes(scene)
        for window in iter_windows(height, width, args.tile_size, args.stride):
            top = int(window.row_off)
            left = int(window.col_off)
            tile_id = f"{scene['stem']}_r{top:05d}_c{left:05d}.tif"
            items.append(
                {
                    "tile_id": tile_id,
                    "region": scene["region"],
                    "scene_id": scene["scene_id"],
                    "top": top,
                    "left": left,
                    "window": window,
                    "scene": scene,
                }
            )
    if not items:
        raise ValueError("No tiles produced; check tile size and source image sizes")
    return items


def assign_splits(items: list[dict[str, object]], train_ratio: float, val_ratio: float, seed: int) -> dict[str, list[dict[str, object]]]:
    rng = random.Random(seed)
    shuffled = items.copy()
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError("Split produces an empty subset; adjust ratios")

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def prepare_dirs(out_root: Path, dry_run: bool) -> None:
    for split in ("train", "val", "test"):
        for sub in ("A", "B", "label"):
            target = out_root / split / sub
            if not dry_run:
                target.mkdir(parents=True, exist_ok=True)


def clean_output_root(out_root: Path, overwrite: bool, dry_run: bool) -> None:
    if not out_root.exists():
        return
    if not overwrite:
        return
    if dry_run:
        return
    shutil.rmtree(out_root)


def _read_window(ds: rasterio.io.DatasetReader, window: Window) -> np.ndarray:
    return ds.read(1, window=window)


def _valid_ratio(arr_a: np.ndarray, nodata_a: float | int | None, arr_b: np.ndarray, nodata_b: float | int | None) -> float:
    valid = np.isfinite(arr_a) & np.isfinite(arr_b)
    if nodata_a is not None:
        valid &= arr_a != nodata_a
    if nodata_b is not None:
        valid &= arr_b != nodata_b
    return float(valid.mean())


def write_tile(
    ds: rasterio.io.DatasetReader,
    arr: np.ndarray,
    out_path: Path,
    window: Window,
    dtype: str,
    nodata,
    dry_run: bool,
) -> None:
    if dry_run:
        return
    meta = ds.meta.copy()
    meta.update(
        {
            "driver": "GTiff",
            "height": int(window.height),
            "width": int(window.width),
            "count": 1,
            "dtype": dtype,
            "transform": rasterio.windows.transform(window, ds.transform),
            "compress": "LZW",
            "nodata": nodata,
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **meta) as out_ds:
        out_ds.write(arr, 1)


def write_split(
    out_root: Path,
    split: str,
    items: list[dict[str, object]],
    dry_run: bool,
    strict: bool,
) -> list[dict[str, object]]:
    manifest_rows: list[dict[str, object]] = []
    for item in items:
        scene = item["scene"]
        window = item["window"]
        tile_id = item["tile_id"]
        with rasterio.open(scene["A"]) as ds_a, rasterio.open(scene["B"]) as ds_b, rasterio.open(scene["label"]) as ds_l:
            arr_a = _read_window(ds_a, window).astype(np.float32, copy=False)
            arr_b = _read_window(ds_b, window).astype(np.float32, copy=False)
            label_raw = _read_window(ds_l, window)

            label_tile = (label_raw > 0).astype(np.uint8)
            if strict:
                valid_values = set(np.unique(label_raw).tolist())
                if not valid_values.issubset({0, 255}):
                    raise ValueError(f"Unexpected label values in {scene['scene_id']}: {sorted(valid_values)}")

            valid_ratio = _valid_ratio(arr_a, ds_a.nodata, arr_b, ds_b.nodata)
            pos_ratio = float(label_tile.mean())

            write_tile(ds_a, arr_a, out_root / split / "A" / tile_id, window, "float32", ds_a.nodata, dry_run)
            write_tile(ds_b, arr_b, out_root / split / "B" / tile_id, window, "float32", ds_b.nodata, dry_run)
            write_tile(ds_l, label_tile, out_root / split / "label" / tile_id, window, "uint8", None, dry_run)

        manifest_rows.append(
            {
                "tile_id": tile_id,
                "split": split,
                "region": item["region"],
                "scene_id": item["scene_id"],
                "top": item["top"],
                "left": item["left"],
                "valid_ratio": valid_ratio,
                "pos_ratio": pos_ratio,
            }
        )
    return manifest_rows


def write_report(
    out_root: Path,
    args: argparse.Namespace,
    scenes: list[dict[str, str | Path]],
    splits: dict[str, list[dict[str, object]]],
    manifest_rows: list[dict[str, object]],
    dry_run: bool,
) -> None:
    payload = {
        "source_root": str(args.src_root),
        "output_root": str(args.out_root),
        "mode": "PRO",
        "params": {
            "tile_size": args.tile_size,
            "stride": args.stride,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": 1.0 - args.train_ratio - args.val_ratio,
            "seed": args.seed,
            "strict": args.strict,
            "dry_run": args.dry_run,
        },
        "scenes_total": len(scenes),
        "scene_ids": [str(scene["scene_id"]) for scene in scenes],
        "tile_counts": {
            "candidate_windows": len(manifest_rows),
            "kept": len(manifest_rows),
        },
        "split_counts": {split.title(): len(items) for split, items in splits.items()},
    }
    if dry_run:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    (out_root / "split_report.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out_root / "split_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["tile_id", "split", "region", "scene_id", "top", "left", "valid_ratio", "pos_ratio"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)


def main() -> None:
    args = parse_args()
    ensure_args(args)

    scenes = build_scenes(args.src_root, args.strict)
    items = build_tile_items(args, scenes)
    splits = assign_splits(items, args.train_ratio, args.val_ratio, args.seed)

    clean_output_root(args.out_root, args.overwrite, args.dry_run)
    prepare_dirs(args.out_root, args.dry_run)

    manifest_rows: list[dict[str, object]] = []
    for split, split_items in splits.items():
        manifest_rows.extend(write_split(args.out_root, split, split_items, args.dry_run, args.strict))

    write_report(args.out_root, args, scenes, splits, manifest_rows, args.dry_run)
    counter = Counter({split: len(split_items) for split, split_items in splits.items()})
    print(f"[DONE] total={len(items)} train={counter['train']} val={counter['val']} test={counter['test']}")


if __name__ == "__main__":
    main()
