#!/usr/bin/env python3
"""统计 S1GFloods_CD 训练集输入图像的三通道 mean/std。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

"""
python ChangeDINO-main/scripts/compute_s1gfloods_cd_stats.py \
  --data-root datasets/S1GFloods_CD_DINO \
  --split train \
  --output datasets/S1GFloods_CD_DINO/channel_stats_s1gfloods_train.json
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Compute channel stats for S1GFloods_CD")
    parser.add_argument("--data-root", type=Path, default=Path("datasets/S1GFloods_CD"))
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/S1GFloods_CD/channel_stats_s1gfloods_train.json"),
    )
    parser.add_argument("--max-samples", type=int, default=-1, help="仅处理前 N 个样本；-1 表示全部。")
    return parser.parse_args()


def load_png_rgb(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    return arr.reshape(-1, 3)


def main() -> None:
    args = parse_args()
    split_root = args.data_root / args.split
    img_dirs = [split_root / "A", split_root / "B"]
    paths: list[Path] = []
    for img_dir in img_dirs:
        if not img_dir.is_dir():
            raise FileNotFoundError(f"Missing image directory: {img_dir}")
        paths.extend(sorted(p for p in img_dir.iterdir() if p.is_file()))

    if args.max_samples > 0:
        paths = paths[: args.max_samples]
    if not paths:
        raise ValueError(f"No images found under {split_root}")

    sum_arr = np.zeros(3, dtype=np.float64)
    sumsq_arr = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    for path in paths:
        arr = load_png_rgb(path)
        sum_arr += arr.sum(axis=0)
        sumsq_arr += np.square(arr).sum(axis=0)
        pixel_count += arr.shape[0]

    mean = sum_arr / pixel_count
    var = np.maximum(sumsq_arr / pixel_count - np.square(mean), 0.0)
    std = np.sqrt(var)
    std = np.maximum(std, 1e-6)

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
