#!/usr/bin/env python3
"""为少样本域适应实验构建 GF3 Henan few-shot 扩充样本。"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window

"""
python ChangeDINO-main/scripts/prepare_gf3_henan_fewshot_expand.py \
  --tiles-root datasets/GF3_Henan_CD_infer \
  --label-source datasets/GF3_Henan/GF3_Zhengzhou_label.tif \
  --target-root datasets/S1GFloods_CD_DINO_ \
  --selection-txt datasets/GF3_Henan_CD_infer/gf3_henan_fewshot_20.txt \
  --num-samples 20 \
  --train-count 16 \
  --seed 42 \
  --mode select_and_import \
  --overwrite
"""

VALID_SPLITS = {"train", "val"}
MANIFEST_FIELDS = [
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


@dataclass(frozen=True)
class TileRecord:
    """描述 GF3 Henan 推理切片及其在整景中的窗口位置。"""

    tile_id: str
    split: str
    top: int
    left: int
    height: int
    width: int
    valid_ratio: float
    a_png: str
    b_png: str
    a_tif: str
    b_tif: str
    label_png: str
    label_tif: str

    @property
    def window(self) -> Window:
        return Window(col_off=self.left, row_off=self.top, width=self.width, height=self.height)


@dataclass(frozen=True)
class SelectionEntry:
    """记录 txt 名单中的 split 分配与 tile_id。"""

    split: str
    tile_id: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Prepare GF3 Henan few-shot expansion samples")
    parser.add_argument("--tiles-root", type=Path, default=Path("datasets/GF3_Henan_CD_infer"))
    parser.add_argument("--label-source", type=Path, default=Path("datasets/GF3_Henan/GF3_Zhengzhou_label.tif"))
    parser.add_argument("--target-root", type=Path, default=Path("datasets/S1GFloods_CD_DINO_"))
    parser.add_argument(
        "--selection-txt",
        type=Path,
        default=Path("datasets/GF3_Henan_CD_infer/gf3_henan_fewshot_20.txt"),
    )
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--train-count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        choices=["select", "import", "select_and_import"],
        default="select_and_import",
        help="select: 只生成名单；import: 只读取名单导入；select_and_import: 两步一起执行。",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def ensure_args(args: argparse.Namespace) -> None:
    if not args.tiles_root.is_dir():
        raise FileNotFoundError(f"Tiles root not found: {args.tiles_root}")
    if not args.label_source.is_file():
        raise FileNotFoundError(f"Label source not found: {args.label_source}")
    if not args.target_root.is_dir():
        raise FileNotFoundError(f"Target root not found: {args.target_root}")
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.train_count < 0 or args.train_count > args.num_samples:
        raise ValueError("--train-count must be within [0, num_samples]")


def load_tile_records(tiles_root: Path) -> dict[str, TileRecord]:
    manifest_path = tiles_root / "tile_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing tile manifest: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Tile manifest is empty: {manifest_path}")

    records: dict[str, TileRecord] = {}
    for row in rows:
        tile_id = row["tile_id"]
        records[tile_id] = TileRecord(
            tile_id=tile_id,
            split=row["split"],
            top=int(row["top"]),
            left=int(row["left"]),
            height=int(row["height"]),
            width=int(row["width"]),
            valid_ratio=float(row["valid_ratio"]),
            a_png=row["a_png"],
            b_png=row["b_png"],
            a_tif=row["a_tif"],
            b_tif=row["b_tif"],
            label_png=row.get("label_png", ""),
            label_tif=row.get("label_tif", ""),
        )
    return records


def build_clean_binary_label(arr: np.ndarray) -> np.ndarray:
    """仅把原始值 1 当作洪水前景，0/3 都视为背景。"""
    return np.where(arr == 1, 255, 0).astype(np.uint8)


def classify_tile(arr: np.ndarray) -> tuple[bool, bool]:
    """返回当前窗口是否含洪水(1)以及是否含 nodata(3)。"""
    has_flood = bool(np.any(arr == 1))
    has_nodata = bool(np.any(arr == 3))
    return has_flood, has_nodata


def build_candidate_records(tile_records: dict[str, TileRecord], label_source: Path) -> list[TileRecord]:
    """筛选出满足“有 1 且无 3”的 GF3 切片。"""
    candidates: list[TileRecord] = []
    with rasterio.open(label_source) as ds_label:
        for record in tile_records.values():
            arr = ds_label.read(1, window=record.window)
            has_flood, has_nodata = classify_tile(arr)
            if has_flood and not has_nodata:
                candidates.append(record)
    return sorted(candidates, key=lambda item: item.tile_id)


def build_selection_entries(records: list[TileRecord], num_samples: int, train_count: int, seed: int) -> list[SelectionEntry]:
    if len(records) < num_samples:
        raise ValueError(f"Not enough eligible GF3 tiles: need {num_samples}, got {len(records)}")

    rng = random.Random(seed)
    chosen = records.copy()
    rng.shuffle(chosen)
    chosen = chosen[:num_samples]

    entries: list[SelectionEntry] = []
    for index, record in enumerate(chosen):
        split = "train" if index < train_count else "val"
        entries.append(SelectionEntry(split=split, tile_id=record.tile_id))
    return sorted(entries, key=lambda item: (item.split, item.tile_id))


def write_selection_txt(path: Path, entries: list[SelectionEntry], dry_run: bool) -> None:
    lines = [
        "# GF3 Henan few-shot expansion list",
        "# format: <split>\\t<tile_id>",
        *[f"{entry.split}\t{entry.tile_id}" for entry in entries],
        "",
    ]
    if dry_run:
        print("\n".join(lines))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def load_selection_txt(path: Path) -> list[SelectionEntry]:
    if not path.is_file():
        raise FileNotFoundError(f"Selection txt not found: {path}")

    entries: list[SelectionEntry] = []
    for lineno, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"Line {lineno}: expected '<split> <tile_id>', got: {raw_line}")
        split, tile_id = parts
        if split not in VALID_SPLITS:
            raise ValueError(f"Line {lineno}: invalid split '{split}', expected one of {sorted(VALID_SPLITS)}")
        entries.append(SelectionEntry(split=split, tile_id=tile_id))

    if not entries:
        raise ValueError(f"Selection txt is empty: {path}")
    return entries


def validate_selection_entries(
    entries: list[SelectionEntry],
    tile_records: dict[str, TileRecord],
    label_source: Path,
    strict: bool,
    expected_total: int,
    expected_train_count: int,
) -> None:
    seen: set[str] = set()
    with rasterio.open(label_source) as ds_label:
        for entry in entries:
            if entry.tile_id in seen:
                raise ValueError(f"Duplicate tile_id in selection txt: {entry.tile_id}")
            seen.add(entry.tile_id)

            record = tile_records.get(entry.tile_id)
            if record is None:
                raise ValueError(f"Unknown tile_id in selection txt: {entry.tile_id}")

            arr = ds_label.read(1, window=record.window)
            has_flood, has_nodata = classify_tile(arr)
            if not has_flood or has_nodata:
                raise ValueError(
                    f"Tile {entry.tile_id} is not eligible: requires value 1 and forbids value 3 in raw label"
                )

    if strict:
        if len(entries) != expected_total:
            raise ValueError(f"Strict mode expects {expected_total} entries, got {len(entries)}")
        train_count = sum(entry.split == "train" for entry in entries)
        val_count = sum(entry.split == "val" for entry in entries)
        if train_count != expected_train_count or val_count != expected_total - expected_train_count:
            raise ValueError(
                "Strict mode split mismatch: "
                f"expected train={expected_train_count}, val={expected_total - expected_train_count}; "
                f"got train={train_count}, val={val_count}"
            )


def load_manifest_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing manifest: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


def build_existing_ids(rows: list[dict[str, str]]) -> set[str]:
    return {row["sample_id"] for row in rows}


def build_existing_row_map(*row_groups: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    existing: dict[str, dict[str, str]] = {}
    for rows in row_groups:
        for row in rows:
            existing[row["sample_id"]] = row
    return existing


def copy_file(src: Path, dst: Path, overwrite: bool, dry_run: bool) -> None:
    if not src.is_file():
        raise FileNotFoundError(f"Missing source file: {src}")
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"Target file exists, rerun with --overwrite: {dst}")
        if not dry_run:
            dst.unlink()
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def write_png_label(arr: np.ndarray, out_path: Path, overwrite: bool, dry_run: bool) -> None:
    if out_path.exists():
        if not overwrite:
            raise FileExistsError(f"Target file exists, rerun with --overwrite: {out_path}")
        if not dry_run:
            out_path.unlink()
    if dry_run:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="L").save(out_path)


def write_label_tif(
    src_ds: rasterio.io.DatasetReader,
    arr: np.ndarray,
    out_path: Path,
    window: Window,
    overwrite: bool,
    dry_run: bool,
) -> None:
    if out_path.exists():
        if not overwrite:
            raise FileExistsError(f"Target file exists, rerun with --overwrite: {out_path}")
        if not dry_run:
            out_path.unlink()
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
            "nodata": 0,
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as out_ds:
        out_ds.write(arr, 1)


def build_manifest_row(entry: SelectionEntry, record: TileRecord) -> dict[str, str]:
    sample_id = record.tile_id
    split = entry.split
    return {
        "sample_id": sample_id,
        "source": "gf3_henan_fewshot",
        "region": "GF3_Henan",
        "split": split,
        "a_png": str(Path(split) / "A" / f"{sample_id}.png"),
        "b_png": str(Path(split) / "B" / f"{sample_id}.png"),
        "label_png": str(Path(split) / "label" / f"{sample_id}.png"),
        "a_tif": str(Path(f"{split}_tif") / "A" / f"{sample_id}.tif"),
        "b_tif": str(Path(f"{split}_tif") / "B" / f"{sample_id}.tif"),
        "label_tif": str(Path(f"{split}_tif") / "label" / f"{sample_id}.tif"),
        "row_off": str(record.top),
        "col_off": str(record.left),
        "height": str(record.height),
        "width": str(record.width),
        "valid_ratio": f"{record.valid_ratio:.6f}",
    }


def remove_existing_sample_files(row: dict[str, str], target_root: Path, dry_run: bool) -> None:
    """覆盖导入时同步清理旧 split 下的残留文件。"""
    for key in ("a_png", "b_png", "label_png", "a_tif", "b_tif", "label_tif"):
        rel_path = row.get(key, "")
        if not rel_path:
            continue
        path = target_root / rel_path
        if path.exists() and not dry_run:
            path.unlink()


def write_manifest(path: Path, rows: list[dict[str, str]], dry_run: bool) -> None:
    if dry_run:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def update_split_report(
    target_root: Path,
    manifest_train: list[dict[str, str]],
    manifest_val: list[dict[str, str]],
    args: argparse.Namespace,
    imported_entries: list[SelectionEntry],
    dry_run: bool,
) -> None:
    report_path = target_root / "split_report.json"
    if not report_path.is_file():
        return

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    counts = {"train": len(manifest_train), "val": len(manifest_val)}
    counts_by_source = {
        "train": dict(Counter(row["source"] for row in manifest_train)),
        "val": dict(Counter(row["source"] for row in manifest_val)),
    }
    examples = {
        "train": [row["sample_id"] for row in manifest_train[:5]],
        "val": [row["sample_id"] for row in manifest_val[:5]],
    }

    payload["counts"] = counts
    payload["counts_by_source"] = counts_by_source
    payload["examples"] = examples
    payload["gf3_henan_fewshot_expand"] = {
        "tiles_root": str(args.tiles_root),
        "label_source": str(args.label_source),
        "selection_txt": str(args.selection_txt),
        "mode": args.mode,
        "seed": args.seed,
        "imported_count": len(imported_entries),
        "imported_train": sum(entry.split == "train" for entry in imported_entries),
        "imported_val": sum(entry.split == "val" for entry in imported_entries),
        "overwrite": bool(args.overwrite),
        "strict": bool(args.strict),
        "dry_run": bool(dry_run),
    }

    if dry_run:
        print(json.dumps(payload["gf3_henan_fewshot_expand"], indent=2, ensure_ascii=False))
        return
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def import_selection(
    entries: list[SelectionEntry],
    tile_records: dict[str, TileRecord],
    args: argparse.Namespace,
) -> None:
    manifest_train_path = args.target_root / "manifest_train.csv"
    manifest_val_path = args.target_root / "manifest_val.csv"
    manifest_all_path = args.target_root / "manifest_all.csv"

    manifest_train_rows = load_manifest_rows(manifest_train_path)
    manifest_val_rows = load_manifest_rows(manifest_val_path)

    existing_ids = build_existing_ids(manifest_train_rows) | build_existing_ids(manifest_val_rows)
    existing_row_map = build_existing_row_map(manifest_train_rows, manifest_val_rows)
    selected_ids = {entry.tile_id for entry in entries}
    new_train_rows: list[dict[str, str]] = []
    new_val_rows: list[dict[str, str]] = []

    with rasterio.open(args.label_source) as ds_label:
        for entry in entries:
            record = tile_records[entry.tile_id]
            sample_id = record.tile_id

            if sample_id in existing_ids and not args.overwrite:
                raise FileExistsError(f"Sample already exists in target dataset: {sample_id}")
            if sample_id in existing_row_map and args.overwrite:
                remove_existing_sample_files(existing_row_map[sample_id], args.target_root, dry_run=args.dry_run)

            src_a_png = args.tiles_root / record.a_png
            src_b_png = args.tiles_root / record.b_png
            src_a_tif = args.tiles_root / record.a_tif
            src_b_tif = args.tiles_root / record.b_tif

            dst_a_png = args.target_root / entry.split / "A" / f"{sample_id}.png"
            dst_b_png = args.target_root / entry.split / "B" / f"{sample_id}.png"
            dst_label_png = args.target_root / entry.split / "label" / f"{sample_id}.png"
            dst_a_tif = args.target_root / f"{entry.split}_tif" / "A" / f"{sample_id}.tif"
            dst_b_tif = args.target_root / f"{entry.split}_tif" / "B" / f"{sample_id}.tif"
            dst_label_tif = args.target_root / f"{entry.split}_tif" / "label" / f"{sample_id}.tif"

            copy_file(src_a_png, dst_a_png, overwrite=args.overwrite, dry_run=args.dry_run)
            copy_file(src_b_png, dst_b_png, overwrite=args.overwrite, dry_run=args.dry_run)
            copy_file(src_a_tif, dst_a_tif, overwrite=args.overwrite, dry_run=args.dry_run)
            copy_file(src_b_tif, dst_b_tif, overwrite=args.overwrite, dry_run=args.dry_run)

            raw_label = ds_label.read(1, window=record.window)
            label_png = build_clean_binary_label(raw_label)
            write_png_label(label_png, dst_label_png, overwrite=args.overwrite, dry_run=args.dry_run)
            write_label_tif(
                ds_label,
                label_png,
                dst_label_tif,
                record.window,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )

            row = build_manifest_row(entry, record)
            if entry.split == "train":
                new_train_rows.append(row)
            else:
                new_val_rows.append(row)

    manifest_train_without_overwritten = [row for row in manifest_train_rows if row["sample_id"] not in selected_ids]
    manifest_val_without_overwritten = [row for row in manifest_val_rows if row["sample_id"] not in selected_ids]

    final_manifest_train = manifest_train_without_overwritten + new_train_rows
    final_manifest_val = manifest_val_without_overwritten + new_val_rows
    final_manifest_all = final_manifest_train + final_manifest_val

    write_manifest(manifest_train_path, final_manifest_train, dry_run=args.dry_run)
    write_manifest(manifest_val_path, final_manifest_val, dry_run=args.dry_run)
    write_manifest(manifest_all_path, final_manifest_all, dry_run=args.dry_run)
    update_split_report(
        args.target_root,
        manifest_train=final_manifest_train,
        manifest_val=final_manifest_val,
        args=args,
        imported_entries=entries,
        dry_run=args.dry_run,
    )


def summarize_selection(entries: list[SelectionEntry]) -> dict[str, int]:
    return dict(Counter(entry.split for entry in entries))


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.tiles_root = Path(args.tiles_root)
    args.label_source = Path(args.label_source)
    args.target_root = Path(args.target_root)
    args.selection_txt = Path(args.selection_txt)
    ensure_args(args)

    tile_records = load_tile_records(args.tiles_root)
    selected_entries: list[SelectionEntry] | None = None

    if args.mode in {"select", "select_and_import"}:
        candidates = build_candidate_records(tile_records, args.label_source)
        selected_entries = build_selection_entries(
            candidates,
            num_samples=args.num_samples,
            train_count=args.train_count,
            seed=args.seed,
        )
        write_selection_txt(args.selection_txt, selected_entries, dry_run=args.dry_run)
        summary = summarize_selection(selected_entries)
        print(
            "[SELECT] eligible={eligible} chosen={chosen} train={train} val={val} txt={txt}".format(
                eligible=len(candidates),
                chosen=len(selected_entries),
                train=summary.get("train", 0),
                val=summary.get("val", 0),
                txt=args.selection_txt,
            )
        )

    if args.mode in {"import", "select_and_import"}:
        entries = selected_entries if selected_entries is not None else load_selection_txt(args.selection_txt)
        validate_selection_entries(
            entries,
            tile_records=tile_records,
            label_source=args.label_source,
            strict=args.strict,
            expected_total=args.num_samples,
            expected_train_count=args.train_count,
        )
        import_selection(entries, tile_records=tile_records, args=args)
        summary = summarize_selection(entries)
        print(
            "[IMPORT] total={total} train={train} val={val} target={target}".format(
                total=len(entries),
                train=summary.get("train", 0),
                val=summary.get("val", 0),
                target=args.target_root,
            )
        )


if __name__ == "__main__":
    main()
