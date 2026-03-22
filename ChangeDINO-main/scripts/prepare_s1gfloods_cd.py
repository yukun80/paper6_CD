#!/usr/bin/env python3
"""将扁平的 S1GFloods 目录整理为 ChangeDINO 可直接读取的 CD 结构。"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

"""
python ChangeDINO-main/scripts/prepare_s1gfloods_cd.py \
  --src-root datasets/S1GFloods \
  --out-root datasets/S1GFloods_CD_DINO \
  --seed 42 \
  --overwrite
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Prepare S1GFloods in train/val/test CD layout")
    parser.add_argument("--src-root", type=Path, default=Path("datasets/S1GFloods"))
    parser.add_argument("--out-root", type=Path, default=Path("datasets/S1GFloods_CD_DINO"))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--link-mode",
        choices=["copy", "symlink", "hardlink"],
        default="copy",
        help="输出文件写入方式；copy 最稳妥，symlink/hardlink 更省空间。",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def ensure_args(args: argparse.Namespace) -> None:
    if not 0 < args.train_ratio < 1:
        raise ValueError("--train-ratio must be within (0, 1)")
    if not 0 <= args.val_ratio < 1:
        raise ValueError("--val-ratio must be within [0, 1)")
    if args.train_ratio + args.val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1")


def scan_files(root: Path, subdir: str) -> dict[str, Path]:
    folder = root / subdir
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing directory: {folder}")
    files = sorted(p for p in folder.iterdir() if p.is_file())
    return {p.name: p for p in files}


def build_samples(src_root: Path, strict: bool) -> list[dict[str, Path | str]]:
    a_map = scan_files(src_root, "A")
    b_map = scan_files(src_root, "B")
    label_map = scan_files(src_root, "Label")

    names_a = sorted(a_map)
    names_b = sorted(b_map)
    names_l = sorted(label_map)
    if names_a != names_b or names_a != names_l:
        raise ValueError("A/B/Label file names are not aligned")

    samples: list[dict[str, Path | str]] = []
    for name in names_a:
        if strict and Path(name).suffix.lower() != ".png":
            raise ValueError(f"Expected PNG only, got: {name}")
        samples.append({"name": name, "A": a_map[name], "B": b_map[name], "label": label_map[name]})
    return samples


def assign_splits(
    samples: list[dict[str, Path | str]], train_ratio: float, val_ratio: float, seed: int
) -> dict[str, list[dict[str, Path | str]]]:
    rng = random.Random(seed)
    shuffled = samples.copy()
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


def write_one(src: Path, dst: Path, link_mode: str, overwrite: bool, dry_run: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        if not dry_run:
            dst.unlink()
    if dry_run:
        return
    if link_mode == "copy":
        shutil.copy2(src, dst)
    elif link_mode == "symlink":
        dst.symlink_to(src.resolve())
    elif link_mode == "hardlink":
        dst.hardlink_to(src)
    else:
        raise ValueError(f"Unsupported link mode: {link_mode}")


def write_split(
    out_root: Path,
    split: str,
    items: list[dict[str, Path | str]],
    link_mode: str,
    overwrite: bool,
    dry_run: bool,
) -> None:
    for item in items:
        name = str(item["name"])
        write_one(Path(item["A"]), out_root / split / "A" / name, link_mode, overwrite, dry_run)
        write_one(Path(item["B"]), out_root / split / "B" / name, link_mode, overwrite, dry_run)
        write_one(Path(item["label"]), out_root / split / "label" / name, link_mode, overwrite, dry_run)


def write_report(
    out_root: Path,
    args: argparse.Namespace,
    splits: dict[str, list[dict[str, Path | str]]],
    dry_run: bool,
) -> None:
    payload = {
        "src_root": str(args.src_root),
        "out_root": str(args.out_root),
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": 1.0 - args.train_ratio - args.val_ratio,
        "link_mode": args.link_mode,
        "counts": {split: len(items) for split, items in splits.items()},
        "examples": {split: [str(item["name"]) for item in items[:5]] for split, items in splits.items()},
    }
    if dry_run:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    (out_root / "split_report.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_args(args)

    samples = build_samples(args.src_root, args.strict)
    splits = assign_splits(samples, args.train_ratio, args.val_ratio, args.seed)

    prepare_dirs(args.out_root, args.dry_run)
    for split, items in splits.items():
        write_split(args.out_root, split, items, args.link_mode, args.overwrite, args.dry_run)

    write_report(args.out_root, args, splits, args.dry_run)
    counter = Counter({split: len(items) for split, items in splits.items()})
    print(f"[DONE] total={len(samples)} train={counter['train']} val={counter['val']} test={counter['test']}")


if __name__ == "__main__":
    main()
