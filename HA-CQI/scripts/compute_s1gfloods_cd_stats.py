#!/usr/bin/env python3
"""统计 HA-CQI SAR 变化检测数据集的三通道 mean/std。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.runtime_snapshot import (  # noqa: E402
    atomic_write_json,
    compute_stats_for_paths,
    scan_split_files,
)

"""
python HA-CQI/scripts/compute_s1gfloods_cd_stats.py \
  --data-root datasets/S1GFloods_CD_DINO_BG_75_25_ \
  --split train \
  --output datasets/S1GFloods_CD_DINO_BG_75_25_/channel_stats_s1gfloods_train.json
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Compute channel stats for S1GFloods_CD_DINO")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("datasets/S1GFloods_CD_DINO_BG_75_25_"),
    )
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "datasets/S1GFloods_CD_DINO_BG_75_25_/channel_stats_s1gfloods_train.json"
        ),
    )
    parser.add_argument("--max-samples", type=int, default=-1, help="仅处理前 N 个样本；-1 表示全部。")
    return parser.parse_args()


def collect_split_image_paths(data_root: Path, split: str) -> list[Path]:
    """按 A 后 B、各目录文件名字典序返回与 CLI 一致的统计输入。"""
    files = scan_split_files(data_root, split)
    return [files.a[name] for name in files.filenames] + [
        files.b[name] for name in files.filenames
    ]


def compute_stats_payload(
    *,
    data_root: Path,
    split: str,
    paths: Iterable[Path] | None = None,
    max_samples: int = -1,
) -> dict[str, object]:
    """计算统计 payload；允许导入事务对尚未落盘的逻辑路径进行预计算。"""
    selected_paths = list(paths) if paths is not None else collect_split_image_paths(data_root, split)
    if max_samples > 0:
        selected_paths = selected_paths[:max_samples]
    return compute_stats_for_paths(data_root, split, selected_paths)


def write_stats_payload(payload: dict[str, object], output: Path) -> None:
    """在目标目录内原子替换统计文件，避免中断留下半写 JSON。"""
    atomic_write_json(payload, output)


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
