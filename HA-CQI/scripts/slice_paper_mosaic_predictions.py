#!/usr/bin/env python3
"""将 HA-CQI paper 整景预测结果回切为 demo 可用的二值 PNG。"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
REPO_ROOT = PROJECT_ROOT.parent


@dataclass(frozen=True)
class SliceTask:
    """记录一个整景预测结果与对应推理数据集 manifest 的绑定关系。"""

    key: str
    input_tif: Path
    manifest: Path
    tiles_root: Path
    output_dir: Path


DEFAULT_TASKS = {
    "zhengzhou": SliceTask(
        key="zhengzhou",
        input_tif=PROJECT_ROOT / "outputs" / "Zhengzhou_paper" / "Zhengzhou_change_binary_final.tif",
        manifest=REPO_ROOT / "datasets" / "GF3_Henan_CD_infer" / "tile_manifest.csv",
        tiles_root=REPO_ROOT / "datasets" / "GF3_Henan_CD_infer",
        output_dir=PROJECT_ROOT / "outputs" / "Zhengzhou_paper" / "tile_png",
    ),
    "zhuozhou": SliceTask(
        key="zhuozhou",
        input_tif=PROJECT_ROOT / "outputs" / "Zhuozhou_paper" / "Zhuozhou_change_binary_final.tif",
        manifest=REPO_ROOT / "datasets" / "GF3_Zhuozhou_CD_infer" / "tile_manifest.csv",
        tiles_root=REPO_ROOT / "datasets" / "GF3_Zhuozhou_CD_infer",
        output_dir=PROJECT_ROOT / "outputs" / "Zhuozhou_paper" / "tile_png",
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Slice HA-CQI paper full-scene binary predictions into manifest-aligned PNG tiles."
    )
    parser.add_argument(
        "--scene",
        choices=["all", *DEFAULT_TASKS.keys()],
        default="all",
        help="选择要切片的场景；默认同时处理郑州和涿州。",
    )
    return parser


def load_manifest(manifest_path: Path) -> list[dict[str, str]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return sorted(rows, key=lambda row: int(row["tile_index"]))


def ensure_task_inputs(task: SliceTask) -> None:
    if not task.input_tif.is_file():
        raise FileNotFoundError(f"Missing input tif: {task.input_tif}")
    if not task.manifest.is_file():
        raise FileNotFoundError(f"Missing manifest: {task.manifest}")
    if not task.tiles_root.is_dir():
        raise FileNotFoundError(f"Missing tiles root: {task.tiles_root}")


def tile_is_in_bounds(row: dict[str, str], height: int, width: int) -> bool:
    top = int(row["top"])
    left = int(row["left"])
    tile_h = int(row["height"])
    tile_w = int(row["width"])
    return top >= 0 and left >= 0 and top + tile_h <= height and left + tile_w <= width


def read_valid_mask(mask_path: Path) -> np.ndarray:
    with rasterio.open(mask_path) as ds:
        return ds.read(1).astype(np.uint8, copy=False)


def write_binary_png(tile_arr: np.ndarray, valid_mask: np.ndarray, out_path: Path) -> None:
    """按 arr==1 转前景，其他值含 nodata 均写背景；无效区域强制黑色。"""
    binary = ((tile_arr == 1) & (valid_mask > 0)).astype(np.uint8) * 255
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(binary).save(out_path)


def write_report(task: SliceTask, report: dict[str, object]) -> None:
    report_path = task.output_dir.parent / "slice_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def process_task(task: SliceTask) -> dict[str, object]:
    ensure_task_inputs(task)
    rows = load_manifest(task.manifest)
    task.output_dir.mkdir(parents=True, exist_ok=True)

    missing_masks: list[str] = []
    out_of_bounds: list[str] = []
    shape_mismatch: list[dict[str, object]] = []
    written = 0

    with rasterio.open(task.input_tif) as src:
        for row in rows:
            tile_id = row["tile_id"]
            top = int(row["top"])
            left = int(row["left"])
            tile_h = int(row["height"])
            tile_w = int(row["width"])

            if not tile_is_in_bounds(row, src.height, src.width):
                out_of_bounds.append(tile_id)
                continue

            mask_path = task.tiles_root / row["valid_mask"]
            if not mask_path.is_file():
                missing_masks.append(tile_id)
                continue

            tile_arr = src.read(1, window=Window(left, top, tile_w, tile_h))
            valid_mask = read_valid_mask(mask_path)
            if valid_mask.shape != tile_arr.shape:
                shape_mismatch.append(
                    {
                        "tile_id": tile_id,
                        "prediction_shape": list(tile_arr.shape),
                        "valid_mask_shape": list(valid_mask.shape),
                    }
                )
                continue

            write_binary_png(tile_arr, valid_mask, task.output_dir / f"{tile_id}.png")
            written += 1

        report = {
            "scene": task.key,
            "input_tif": str(task.input_tif),
            "manifest": str(task.manifest),
            "tiles_root": str(task.tiles_root),
            "output_dir": str(task.output_dir),
            "source_shape": [src.height, src.width],
            "total_tiles": len(rows),
            "written": written,
            "skipped": len(missing_masks) + len(out_of_bounds) + len(shape_mismatch),
            "missing_mask_count": len(missing_masks),
            "missing_masks": missing_masks,
            "out_of_bounds_count": len(out_of_bounds),
            "out_of_bounds": out_of_bounds,
            "shape_mismatch_count": len(shape_mismatch),
            "shape_mismatch": shape_mismatch,
        }

    write_report(task, report)
    return report


def select_tasks(scene: str) -> list[SliceTask]:
    if scene == "all":
        return [DEFAULT_TASKS["zhengzhou"], DEFAULT_TASKS["zhuozhou"]]
    return [DEFAULT_TASKS[scene]]


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    for task in select_tasks(args.scene):
        report = process_task(task)
        print(
            f"[DONE] {task.key}: total={report['total_tiles']} "
            f"written={report['written']} skipped={report['skipped']} "
            f"output={task.output_dir}"
        )


if __name__ == "__main__":
    main()
