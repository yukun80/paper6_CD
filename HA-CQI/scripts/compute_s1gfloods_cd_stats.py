#!/usr/bin/env python3
"""统计 HA-CQI SAR 变化检测数据集的三通道 mean/std。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import rasterio

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.tif_io import build_valid_mask, stretch_sar_array  # noqa: E402
from utils.provenance import sha256_file  # noqa: E402

"""
python HA-CQI/scripts/compute_s1gfloods_cd_stats.py \
  --data-root datasets/S1GFloods_CD_DINO_BG_75_25 \
  --split train \
  --output datasets/S1GFloods_CD_DINO_BG_75_25/channel_stats_s1gfloods_train.json
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Compute channel stats for S1GFloods_CD_DINO")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("datasets/S1GFloods_CD_DINO_BG_75_25"),
    )
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "datasets/S1GFloods_CD_DINO_BG_75_25/channel_stats_s1gfloods_train.json"
        ),
    )
    parser.add_argument("--max-samples", type=int, default=-1, help="仅处理前 N 个样本；-1 表示全部。")
    return parser.parse_args()


def load_png_rgb(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    return arr.reshape(-1, 3)


def load_tif_rgb(path: Path) -> np.ndarray:
    """按训练时的 SAR tif 读取方式转成 3 通道 [0, 1] 数组。"""
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype(np.float32, copy=False)
        valid_mask = build_valid_mask(arr, ds.nodata)
    stretched = stretch_sar_array(arr, valid_mask)
    rgb = np.repeat(stretched[:, :, None], 3, axis=2).astype(np.float64, copy=False)
    return rgb.reshape(-1, 3)


def load_image_rgb(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".tif", ".tiff"}:
        return load_tif_rgb(path)
    return load_png_rgb(path)


def load_dataset_fingerprint(data_root: Path) -> str | None:
    report_path = data_root / "split_report.json"
    if not report_path.is_file():
        return None
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    value = payload.get("dataset_fingerprint")
    return str(value) if value else None


def main() -> None:
    args = parse_args()
    split_root = args.data_root / args.split
    img_dirs = [split_root / "A", split_root / "B"]
    paths: list[Path] = []
    for img_dir in img_dirs:
        if not img_dir.is_dir():
            raise FileNotFoundError(f"Missing image directory: {img_dir}")
        paths.extend(
            sorted(
                p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
            )
        )

    if args.max_samples > 0:
        paths = paths[: args.max_samples]
    if not paths:
        raise ValueError(f"No images found under {split_root}")

    sum_arr = np.zeros(3, dtype=np.float64)
    sumsq_arr = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    for path in paths:
        arr = load_image_rgb(path)
        sum_arr += arr.sum(axis=0)
        sumsq_arr += np.square(arr).sum(axis=0)
        pixel_count += arr.shape[0]

    mean = sum_arr / pixel_count
    var = np.maximum(sumsq_arr / pixel_count - np.square(mean), 0.0)
    std = np.sqrt(var)
    std = np.maximum(std, 1e-6)

    payload = {
        "format_version": 2,
        "data_root": str(args.data_root),
        "split": args.split,
        "dataset_fingerprint": load_dataset_fingerprint(args.data_root),
        "manifest_sha256": (
            sha256_file(args.data_root / f"manifest_{args.split}.csv")
            if (args.data_root / f"manifest_{args.split}.csv").is_file()
            else None
        ),
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
