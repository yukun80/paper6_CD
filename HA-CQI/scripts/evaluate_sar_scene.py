#!/usr/bin/env python3
"""对 HA-CQI 整景概率图执行 nodata-aware 洪水二值评估。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import rasterio

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.tif_io import build_valid_mask  # noqa: E402
from utils.flood_evaluation import (  # noqa: E402
    FloodEvaluationAccumulator,
    build_threshold_grid,
)


def atomic_json_save(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=None,
        help="整景输出目录；自动读取 mosaic/change_prob.tif 与可选 change_binary.tif。",
    )
    parser.add_argument("--probability", type=Path, default=None)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--filtered-binary", type=Path, default=None)
    parser.add_argument("--valid-mask", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--tiny-area-thresh", type=int, default=100)
    parser.add_argument("--small-area-thresh", type=int, default=400)
    parser.add_argument("--report", type=Path, default=None)
    return parser


def resolve_inputs(args: argparse.Namespace) -> argparse.Namespace:
    if args.prediction_dir is not None:
        prediction_dir = args.prediction_dir.expanduser().resolve()
        mosaic_dir = prediction_dir / "mosaic"
        if args.probability is None:
            args.probability = mosaic_dir / "change_prob.tif"
        if args.filtered_binary is None:
            candidate = mosaic_dir / "change_binary.tif"
            if candidate.is_file():
                args.filtered_binary = candidate
        if args.threshold is None:
            infer_report = prediction_dir / "infer_report.json"
            if infer_report.is_file():
                payload = json.loads(infer_report.read_text(encoding="utf-8"))
                if payload.get("threshold") is not None:
                    args.threshold = float(payload["threshold"])
                    args.threshold_source = "infer_report"
        if args.report is None:
            args.report = prediction_dir / "scene_evaluation.json"
    if args.probability is None:
        raise ValueError("Provide --prediction-dir or --probability")
    args.probability = args.probability.expanduser().resolve()
    args.ground_truth = args.ground_truth.expanduser().resolve()
    if args.filtered_binary is not None:
        args.filtered_binary = args.filtered_binary.expanduser().resolve()
    if args.valid_mask is not None:
        args.valid_mask = args.valid_mask.expanduser().resolve()
    if args.report is None:
        args.report = args.probability.with_name("scene_evaluation.json")
    else:
        args.report = args.report.expanduser().resolve()
    if args.threshold is None:
        raise ValueError(
            "Evaluation requires --threshold or prediction-dir/infer_report.json threshold"
        )
    elif not hasattr(args, "threshold_source"):
        args.threshold_source = "explicit_cli"
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be within [0, 1]")
    for path in (args.probability, args.ground_truth, args.filtered_binary, args.valid_mask):
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)
    return args


def read_aligned(
    path: Path,
    reference: rasterio.io.DatasetReader | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    with rasterio.open(path) as dataset:
        if reference is not None:
            if (dataset.width, dataset.height) != (reference.width, reference.height):
                raise ValueError(f"Raster shape mismatch: {path}")
            if dataset.crs != reference.crs or not dataset.transform.almost_equals(
                reference.transform
            ):
                raise ValueError(f"Raster grid mismatch: {path}")
        array = dataset.read(1)
        valid = build_valid_mask(array, dataset.nodata)
        meta = {
            "path": str(path),
            "shape": [int(dataset.height), int(dataset.width)],
            "crs": str(dataset.crs) if dataset.crs else None,
            "transform": str(dataset.transform),
            "dtype": dataset.dtypes[0],
            "nodata": dataset.nodata,
        }
    return array, valid, meta


def evaluate_probability(
    probabilities: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    thresholds: list[float],
    reference_threshold: float,
    args: argparse.Namespace,
) -> dict[str, object]:
    evaluator = FloodEvaluationAccumulator(
        thresholds,
        reference_threshold=reference_threshold,
        tiny_area_thresh=args.tiny_area_thresh,
        small_area_thresh=args.small_area_thresh,
    )
    evaluator.update(probabilities, target, valid)
    result = evaluator.scores()
    result["reference"]["ignored_pixels"] = int(valid.size - np.count_nonzero(valid))
    return result


def main() -> None:
    args = resolve_inputs(build_parser().parse_args())
    with rasterio.open(args.probability) as probability_dataset:
        probabilities = probability_dataset.read(1).astype(np.float32, copy=False)
        probability_valid = build_valid_mask(probabilities, probability_dataset.nodata)
        probability_meta = {
            "path": str(args.probability),
            "shape": [probability_dataset.height, probability_dataset.width],
            "crs": str(probability_dataset.crs) if probability_dataset.crs else None,
            "transform": str(probability_dataset.transform),
            "dtype": probability_dataset.dtypes[0],
            "nodata": probability_dataset.nodata,
        }
        ground_truth, gt_valid, gt_meta = read_aligned(
            args.ground_truth, probability_dataset
        )
        # 历史 GF-3 原始标签以 3 表示 nodata；即使 metadata 丢失也不得视作洪水。
        gt_valid &= ground_truth != 3
        valid = probability_valid & gt_valid
        if args.valid_mask is not None:
            external_mask, mask_valid, _ = read_aligned(args.valid_mask, probability_dataset)
            valid &= mask_valid & (external_mask > 0)
        target = (ground_truth > 0) & valid

        thresholds = build_threshold_grid(
            args.threshold_min, args.threshold_max, args.threshold_step
        )
        raw = evaluate_probability(
            probabilities,
            target,
            valid,
            thresholds,
            float(args.threshold),
            args,
        )

        filtered = None
        filtered_meta = None
        if args.filtered_binary is not None:
            filtered_array, filtered_valid, filtered_meta = read_aligned(
                args.filtered_binary, probability_dataset
            )
            filtered_mask = filtered_array > 0
            filtered = evaluate_probability(
                filtered_mask.astype(np.float32),
                target,
                valid & filtered_valid,
                [0.5],
                0.5,
                args,
            )

    report = {
        "format_version": 1,
        "threshold": float(args.threshold),
        "threshold_source": str(args.threshold_source),
        "valid_pixels": int(np.count_nonzero(valid)),
        "ignored_pixels": int(valid.size - np.count_nonzero(valid)),
        "rasters": {
            "probability": probability_meta,
            "ground_truth": gt_meta,
            "filtered_binary": filtered_meta,
        },
        "raw": raw,
        "filtered": filtered,
    }
    atomic_json_save(report, args.report)
    reference = raw["reference"]
    print(
        f"raw@{args.threshold:.2f} | IoU={reference['iou_1']:.6f} "
        f"F1={reference['F1_1']:.6f} P={reference['precision_1']:.6f} "
        f"R={reference['recall_1']:.6f} ignored={report['ignored_pixels']}"
    )
    if filtered is not None:
        filtered_reference = filtered["reference"]
        print(
            f"filtered | IoU={filtered_reference['iou_1']:.6f} "
            f"F1={filtered_reference['F1_1']:.6f} "
            f"P={filtered_reference['precision_1']:.6f} "
            f"R={filtered_reference['recall_1']:.6f}"
        )
    print(f"report={args.report}")


if __name__ == "__main__":
    main()
