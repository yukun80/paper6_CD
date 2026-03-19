#!/usr/bin/env python3
"""统计 VarFloods_CD 训练集输入 tif 的三通道 mean/std。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.tif_io import read_sar_tif

"""
python ChangeDINO-main/scripts/compute_varfloods_cd_stats.py \
  --data-root datasets/VarFloods_CD \
  --split train \
  --output datasets/VarFloods_CD/channel_stats_varfloods_train.json
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Compute channel stats for VarFloods_CD")
    parser.add_argument("--data-root", type=Path, default=Path("datasets/VarFloods_CD"))
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/VarFloods_CD/channel_stats_varfloods_train.json"),
    )
    parser.add_argument("--max-samples", type=int, default=-1, help="仅处理前 N 个样本；-1 表示全部。")
    return parser.parse_args()


def collect_tif_paths(split_root: Path) -> list[Path]:
    paths: list[Path] = []
    for subdir in ("A", "B"):
        img_dir = split_root / subdir
        if not img_dir.is_dir():
            raise FileNotFoundError(f"Missing image directory: {img_dir}")
        paths.extend(sorted(p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}))
    return paths


def main() -> None:
    args = parse_args()
    split_root = args.data_root / args.split
    paths = collect_tif_paths(split_root)
    if args.max_samples > 0:
        paths = paths[: args.max_samples]
    if not paths:
        raise ValueError(f"No tif images found under {split_root}")

    sum_arr = np.zeros(3, dtype=np.float64)
    sumsq_arr = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    for path in paths:
        arr = read_sar_tif(path).numpy().astype(np.float64, copy=False)
        arr = np.moveaxis(arr, 0, -1).reshape(-1, 3)
        sum_arr += arr.sum(axis=0)
        sumsq_arr += np.square(arr).sum(axis=0)
        pixel_count += arr.shape[0]

    mean = sum_arr / pixel_count
    var = np.maximum(sumsq_arr / pixel_count - np.square(mean), 0.0)
    std = np.maximum(np.sqrt(var), 1e-6)
    payload = {
        "data_root": str(args.data_root),
        "split": args.split,
        "num_images": len(paths),
        "pixel_count": int(pixel_count),
        "recommended_config_fields": {
            "mean": [float(x) for x in mean.tolist()],
            "std": [float(x) for x in std.tolist()],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[DONE] mean:", payload["recommended_config_fields"]["mean"])
    print("[DONE] std :", payload["recommended_config_fields"]["std"])
    print("[DONE] output:", args.output)


if __name__ == "__main__":
    main()
