#!/usr/bin/env python3
"""统计 HA-CQI SAR 变化检测数据集的三通道 mean/std。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

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


def collect_split_image_paths(data_root: Path, split: str) -> list[Path]:
    """按 A 后 B、各目录文件名字典序返回与 CLI 一致的统计输入。"""
    split_root = data_root / split
    paths: list[Path] = []
    for img_dir in (split_root / "A", split_root / "B"):
        if not img_dir.is_dir():
            raise FileNotFoundError(f"Missing image directory: {img_dir}")
        paths.extend(
            sorted(
                path
                for path in img_dir.iterdir()
                if path.is_file()
                and path.suffix.lower()
                in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
            )
        )
    return paths


def compute_stats_payload(
    *,
    data_root: Path,
    split: str,
    paths: Iterable[Path] | None = None,
    max_samples: int = -1,
    dataset_fingerprint: str | None = None,
    manifest_sha256: str | None = None,
) -> dict[str, object]:
    """计算统计 payload；允许导入事务对尚未落盘的逻辑路径进行预计算。"""
    selected_paths = list(paths) if paths is not None else collect_split_image_paths(data_root, split)
    if max_samples > 0:
        selected_paths = selected_paths[:max_samples]
    if not selected_paths:
        raise ValueError(f"No images found for split={split}: {data_root}")

    sum_arr = np.zeros(3, dtype=np.float64)
    sumsq_arr = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    for path in selected_paths:
        arr = load_image_rgb(path)
        sum_arr += arr.sum(axis=0)
        sumsq_arr += np.square(arr).sum(axis=0)
        pixel_count += arr.shape[0]

    mean = sum_arr / pixel_count
    var = np.maximum(sumsq_arr / pixel_count - np.square(mean), 0.0)
    std = np.maximum(np.sqrt(var), 1e-6)
    manifest_path = data_root / f"manifest_{split}.csv"
    fingerprint = (
        dataset_fingerprint
        if dataset_fingerprint is not None
        else load_dataset_fingerprint(data_root)
    )
    manifest_hash = manifest_sha256
    if manifest_hash is None and manifest_path.is_file():
        manifest_hash = sha256_file(manifest_path)

    return {
        "format_version": 2,
        "data_root": str(data_root),
        "split": split,
        "dataset_fingerprint": fingerprint,
        "manifest_sha256": manifest_hash,
        "num_images": len(selected_paths),
        "pixel_count": int(pixel_count),
        "recommended_config_fields": {
            "mean": [float(x) for x in mean.tolist()],
            "std": [float(x) for x in std.tolist()],
        },
    }


def write_stats_payload(payload: dict[str, object], output: Path) -> None:
    """在目标目录内原子替换统计文件，避免中断留下半写 JSON。"""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, output)


def main() -> None:
    args = parse_args()
    payload = compute_stats_payload(
        data_root=args.data_root,
        split=args.split,
        max_samples=args.max_samples,
    )
    write_stats_payload(payload, args.output)
    print("[DONE] mean:", payload["recommended_config_fields"]["mean"])
    print("[DONE] std :", payload["recommended_config_fields"]["std"])
    print("[DONE] output:", args.output)


if __name__ == "__main__":
    main()
