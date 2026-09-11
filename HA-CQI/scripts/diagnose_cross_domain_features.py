#!/usr/bin/env python3
"""只读审计跨域 FP 的辐射、纹理、DINO 与 HA-CQI 多尺度表征。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from torchvision.transforms.functional import pil_to_tensor

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.tif_io import (  # noqa: E402
    read_binary_label_with_valid_mask,
    read_raster_valid_mask,
)
from utils.prediction import threshold_probability
from model.modules.dino_adapter import DinoV3FeatureExtractor  # noqa: E402
from scripts.infer_sar_scene_tiles import (  # noqa: E402
    load_model,
    parse_and_prepare,
    resolve_existing_path,
)
from utils.flood_evaluation import FloodEvaluationAccumulator  # noqa: E402


DINO_DIAGNOSTIC_LAYERS = (2, 5, 8, 11)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiles-root", type=Path, required=True)
    parser.add_argument("--probability", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument(
        "--role",
        choices=["source_validation", "calibration", "locked_test", "qualitative"],
        required=True,
        help="只有 source_validation/calibration 报告可参与模型选择。",
    )
    parser.add_argument("--ground-truth", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--dino-weight",
        type=Path,
        default=PROJECT_ROOT
        / "dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    )
    parser.add_argument("--dino-arch", default="dinov3_vits16")
    parser.add_argument("--stats-file", type=Path, default=None)
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--max-per-class", type=int, default=8)
    parser.add_argument("--min-valid-ratio", type=float, default=0.50)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _read_manifest(tiles_root: Path) -> list[dict[str, str]]:
    path = tiles_root / "tile_manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing tile manifest: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty tile manifest: {path}")
    return sorted(rows, key=lambda row: int(row["tile_index"]))


def _read_raster(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(path) as dataset:
        array = dataset.read(1).astype(np.float32, copy=False)
        valid = read_raster_valid_mask(dataset, array)
    return array, valid


def _read_tile_target(row: dict[str, str], tiles_root: Path, height: int, width: int) -> np.ndarray:
    """优先读取带源有效性信息的 TIFF，旧 PNG 仍按 0/255 洪水编码兼容。"""
    label_path = tiles_root / (row.get("label_tif") or row["label_png"])
    target, _ = read_binary_label_with_valid_mask(label_path)
    return target[:height, :width].astype(bool)


def _read_tile_coverage(row: dict[str, str], tiles_root: Path, height: int, width: int) -> np.ndarray:
    """融合清单掩膜与可用 SAR TIFF 覆盖区，GT 未知不缩小评估分母。"""
    values, mask_valid = _read_raster(tiles_root / row["valid_mask"])
    valid = mask_valid[:height, :width] & (values[:height, :width] > 0)
    for key in ("a_tif", "b_tif"):
        if row.get(key):
            _, image_valid = _read_raster(tiles_root / row[key])
            valid &= image_valid[:height, :width]
    return valid


def _rectangles_overlap(first: dict[str, object], second: dict[str, object]) -> bool:
    return not (
        int(first["bottom"]) <= int(second["top"])
        or int(second["bottom"]) <= int(first["top"])
        or int(first["right"]) <= int(second["left"])
        or int(second["right"]) <= int(first["left"])
    )


def _select_non_overlapping(
    candidates: list[dict[str, object]],
    count: int,
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for item in sorted(candidates, key=lambda value: float(value["score"]), reverse=True):
        if any(_rectangles_overlap(item, previous) for previous in selected):
            continue
        selected.append(item)
        if len(selected) >= count:
            break
    return selected


def _categorize_tiles(
    rows: list[dict[str, str]],
    tiles_root: Path,
    probability: np.ndarray,
    probability_valid: np.ndarray,
    threshold: float,
    min_valid_ratio: float,
    max_per_class: int,
    has_labels: bool,
) -> dict[str, list[dict[str, object]]]:
    candidates: dict[str, list[dict[str, object]]] = defaultdict(list)
    for index, row in enumerate(rows):
        if float(row.get("valid_ratio", 1.0)) < min_valid_ratio:
            continue
        top, left = int(row["top"]), int(row["left"])
        height, width = int(row["height"]), int(row["width"])
        prob = probability[top : top + height, left : left + width]
        valid = _read_tile_coverage(row, tiles_root, height, width)
        valid &= probability_valid[top : top + height, left : left + width] & np.isfinite(prob)
        valid_count = int(np.count_nonzero(valid))
        if valid_count == 0:
            continue
        pred = threshold_probability(prob, threshold)
        base = {
            "row_index": index,
            "tile_id": row["tile_id"],
            "top": top,
            "left": left,
            "bottom": top + height,
            "right": left + width,
            "mean_probability": float(np.mean(prob[valid])),
        }
        if not has_labels:
            category = "high_prediction" if np.any(pred & valid) else "clean_prediction"
            score = float(np.mean(prob[valid])) if category == "high_prediction" else float(
                1.0 - np.mean(prob[valid])
            )
        else:
            target = _read_tile_target(row, tiles_root, height, width)
            tp = int(np.count_nonzero(pred & target & valid))
            fp = int(np.count_nonzero(pred & ~target & valid))
            gt = int(np.count_nonzero(target & valid))
            if gt > 0 and tp >= fp:
                category = "tp"
                score = tp / max(gt, 1)
            elif fp > 0:
                category = "fp"
                score = fp / max(int(np.count_nonzero(~target & valid)), 1)
            else:
                category = "tn"
                score = float(1.0 - np.mean(prob[~target & valid]))
            base.update({"tp_pixels": tp, "fp_pixels": fp, "gt_pixels": gt})
        base["category"] = category
        base["score"] = float(score)
        candidates[category].append(base)
    return {
        category: _select_non_overlapping(items, max_per_class)
        for category, items in candidates.items()
    }


def _weighted_local_ncc(
    first: np.ndarray,
    second: np.ndarray,
    valid: np.ndarray,
    kernel_size: int = 15,
) -> float:
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    weights = valid.astype(np.float32)
    count = ndimage.convolve(weights, kernel, mode="constant", cval=0.0)
    safe_count = np.maximum(count, 1.0)
    first_sum = ndimage.convolve(first * weights, kernel, mode="constant", cval=0.0)
    second_sum = ndimage.convolve(second * weights, kernel, mode="constant", cval=0.0)
    first_mean = first_sum / safe_count
    second_mean = second_sum / safe_count
    cross_moment = ndimage.convolve(
        first * second * weights,
        kernel,
        mode="constant",
        cval=0.0,
    ) / safe_count
    first_second_moment = ndimage.convolve(
        np.square(first) * weights,
        kernel,
        mode="constant",
        cval=0.0,
    ) / safe_count
    second_second_moment = ndimage.convolve(
        np.square(second) * weights,
        kernel,
        mode="constant",
        cval=0.0,
    ) / safe_count
    covariance = cross_moment - first_mean * second_mean
    first_variance = np.maximum(first_second_moment - np.square(first_mean), 0.0)
    second_variance = np.maximum(second_second_moment - np.square(second_mean), 0.0)
    eligible = valid & (count >= kernel_size * kernel_size * 0.5)
    denominator = np.sqrt(np.maximum(first_variance * second_variance, 1e-8))
    values = covariance / denominator
    return float(np.mean(values[eligible])) if np.any(eligible) else float("nan")


def _gradient_metrics(
    first: np.ndarray,
    second: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float]:
    first_dx = ndimage.sobel(first, axis=1, mode="nearest")
    first_dy = ndimage.sobel(first, axis=0, mode="nearest")
    second_dx = ndimage.sobel(second, axis=1, mode="nearest")
    second_dy = ndimage.sobel(second, axis=0, mode="nearest")
    first_mag = np.hypot(first_dx, first_dy)
    second_mag = np.hypot(second_dx, second_dy)
    eligible = ndimage.binary_erosion(valid, structure=np.ones((3, 3), dtype=bool))
    cosine = (first_dx * second_dx + first_dy * second_dy) / np.maximum(
        first_mag * second_mag,
        1e-6,
    )
    magnitude_change = np.abs(second_mag - first_mag) / np.maximum(
        second_mag + first_mag,
        1e-6,
    )
    if not np.any(eligible):
        return {"gradient_cosine": float("nan"), "gradient_magnitude_change": float("nan")}
    return {
        "gradient_cosine": float(np.mean(cosine[eligible])),
        "gradient_magnitude_change": float(np.mean(magnitude_change[eligible])),
    }


def _cosine_distance(
    first: torch.Tensor,
    second: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> float:
    first_norm = F.normalize(first.float(), dim=1, eps=1e-6)
    second_norm = F.normalize(second.float(), dim=1, eps=1e-6)
    distance = 1.0 - torch.sum(first_norm * second_norm, dim=1)
    if valid_mask is not None:
        mask = F.interpolate(
            valid_mask.float(),
            size=distance.shape[-2:],
            mode="nearest",
        )[:, 0] > 0.5
        distance = distance[mask]
    return float(distance.mean().detach().cpu()) if distance.numel() else float("nan")


def _load_png_tensor(path: Path, device: torch.device) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    return pil_to_tensor(image).float().div_(255.0).unsqueeze(0).to(device)


def _mean_report(values: dict[str, list[float]]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for key, items in sorted(values.items()):
        array = np.asarray(items, dtype=np.float64)
        finite = array[np.isfinite(array)]
        result[key] = float(finite.mean()) if finite.size else None
    return result


def _full_scene_evaluation(
    probability: np.ndarray,
    probability_valid: np.ndarray,
    ground_truth: Path,
    threshold: float,
) -> dict[str, object]:
    target, _ = read_binary_label_with_valid_mask(ground_truth)
    if target.shape != probability.shape:
        raise ValueError(
            f"Full-scene probability/GT shape mismatch: {probability.shape} != {target.shape}"
        )
    evaluator = FloodEvaluationAccumulator([threshold], reference_threshold=threshold)
    evaluator.update(probability, target, probability_valid)
    result = evaluator.scores()
    return {
        "reference": result["reference"],
    }


def _tile_evaluation(
    rows: list[dict[str, str]],
    tiles_root: Path,
    probability: np.ndarray,
    probability_valid: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    """按 manifest tile 统计背景 FP；重叠像素只影响汇总 IoU，不影响 tile FP 定义。"""
    evaluator = FloodEvaluationAccumulator([threshold], reference_threshold=threshold)
    for row in rows:
        top, left = int(row["top"]), int(row["left"])
        height, width = int(row["height"]), int(row["width"])
        prob = probability[top : top + height, left : left + width]
        target = _read_tile_target(row, tiles_root, height, width)
        valid = _read_tile_coverage(row, tiles_root, height, width)
        valid &= probability_valid[top : top + height, left : left + width]
        evaluator.update(prob, target, valid)
    result = evaluator.scores()
    return {
        "note": "overlap pixels repeat across tiles; use this section for tile-level FP diagnostics",
        "reference": result["reference"],
    }


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be within [0, 1]")
    if args.max_per_class < 1:
        raise ValueError("--max-per-class must be positive")
    if not 0.0 <= args.min_valid_ratio <= 1.0:
        raise ValueError("--min-valid-ratio must be within [0, 1]")

    tiles_root = resolve_existing_path(args.tiles_root, expect_file=False)
    probability_path = resolve_existing_path(args.probability, expect_file=True)
    output_path = args.output.expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite explicitly: {output_path}")
    rows = _read_manifest(tiles_root)
    probability, probability_valid = _read_raster(probability_path)
    has_labels = all(row.get("label_tif") or row.get("label_png") for row in rows) and args.role != "qualitative"
    selections = _categorize_tiles(
        rows,
        tiles_root,
        probability,
        probability_valid,
        args.threshold,
        args.min_valid_ratio,
        args.max_per_class,
        has_labels,
    )

    engine = None
    model_opt = None
    if args.checkpoint is not None:
        infer_args = [
            "--tiles-root",
            str(tiles_root),
            "--checkpoint",
            str(args.checkpoint),
            "--threshold",
            str(args.threshold),
            "--gpu_ids",
            str(args.gpu_ids),
            "--skip-tiles",
            "--output-dir",
            str(output_path.parent / ".diagnostic_runtime"),
        ]
        if args.stats_file is not None:
            infer_args.extend(["--stats_file", str(args.stats_file)])
        model_opt, checkpoint_payload = parse_and_prepare(
            infer_args,
            create_output_dirs=False,
        )
        engine = load_model(model_opt, checkpoint_payload)
        extractor = engine.model.encoder.dino_extractor
        device = engine.device
    else:
        device = torch.device(
            f"cuda:{str(args.gpu_ids).split(',')[0]}"
            if torch.cuda.is_available() and str(args.gpu_ids) != "-1"
            else "cpu"
        )
        extractor = DinoV3FeatureExtractor(
            dino_arch=args.dino_arch,
            weights_path=str(resolve_existing_path(args.dino_weight, expect_file=True)),
            fusion_layers=[5, 8, 11],
            device=str(device),
        ).eval()

    per_category: dict[str, dict[str, list[float]]] = {}
    row_by_index = {index: row for index, row in enumerate(rows)}
    for category, selected in selections.items():
        values: dict[str, list[float]] = defaultdict(list)
        for item in selected:
            row = row_by_index[int(item["row_index"])]
            pre_raw, pre_valid = _read_raster(tiles_root / row["a_tif"])
            post_raw, post_valid = _read_raster(tiles_root / row["b_tif"])
            valid_values, mask_valid = _read_raster(tiles_root / row["valid_mask"])
            valid = pre_valid & post_valid & mask_valid & (valid_values > 0)
            values["local_ncc_15"].append(
                _weighted_local_ncc(pre_raw, post_raw, valid, kernel_size=15)
            )
            for name, value in _gradient_metrics(pre_raw, post_raw, valid).items():
                values[name].append(value)

            pre_image = _load_png_tensor(tiles_root / row["a_png"], device)
            post_image = _load_png_tensor(tiles_root / row["b_png"], device)
            valid_tensor = torch.from_numpy(valid.astype(np.float32)).view(
                1, 1, *valid.shape
            ).to(device)
            dino_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            dino_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            with torch.inference_mode():
                pre_dino = extractor.extract_layers(
                    (pre_image - dino_mean) / dino_std,
                    list(DINO_DIAGNOSTIC_LAYERS),
                )
                post_dino = extractor.extract_layers(
                    (post_image - dino_mean) / dino_std,
                    list(DINO_DIAGNOSTIC_LAYERS),
                )
                for layer, pre_feature, post_feature in zip(
                    DINO_DIAGNOSTIC_LAYERS,
                    pre_dino,
                    post_dino,
                ):
                    values[f"dino_layer{layer}_temporal_distance"].append(
                        _cosine_distance(pre_feature, post_feature, valid_tensor)
                    )

                if engine is not None and model_opt is not None:
                    mean = torch.tensor(model_opt.mean, device=device).view(1, 3, 1, 1)
                    std = torch.tensor(model_opt.std, device=device).view(1, 3, 1, 1)
                    context = (
                        torch.amp.autocast(
                            "cuda",
                            dtype=torch.bfloat16
                            if model_opt.amp_dtype == "bf16"
                            else torch.float16,
                        )
                        if device.type == "cuda" and model_opt.amp
                        else nullcontext()
                    )
                    with context:
                        bundle = engine.model.extract_change_features(
                            (pre_image - mean) / std,
                            (post_image - mean) / std,
                        )
                    for level, (pre_feature, post_feature) in enumerate(
                        zip(bundle["aligned_pre"], bundle["aligned_post"]),
                        start=1,
                    ):
                        values[f"p{level}_temporal_distance"].append(
                            _cosine_distance(pre_feature, post_feature, valid_tensor)
                        )
                    for level, feature in enumerate(bundle["change_primitives"], start=1):
                        values[f"p{level}_change_rms"].append(
                            float(feature.float().square().mean().sqrt().cpu())
                        )
        per_category[category] = values

    report: dict[str, object] = {
        "schema_version": 1,
        "label_policy": "unknown_gt_as_background_within_valid_coverage",
        "role": args.role,
        "eligible_for_model_selection": args.role in {"source_validation", "calibration"},
        "tiles_root": str(tiles_root),
        "probability": str(probability_path),
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "threshold": float(args.threshold),
        "dino_diagnostic_layers": list(DINO_DIAGNOSTIC_LAYERS),
        "selection": {
            category: [{key: value for key, value in item.items() if key != "row_index"} for item in items]
            for category, items in selections.items()
        },
        "feature_summary": {
            category: _mean_report(values) for category, values in per_category.items()
        },
    }
    if has_labels:
        report["tile_evaluation"] = _tile_evaluation(
            rows,
            tiles_root,
            probability,
            probability_valid,
            args.threshold,
        )
    if args.ground_truth is not None:
        ground_truth = resolve_existing_path(args.ground_truth, expect_file=True)
        report["ground_truth"] = str(ground_truth)
        report["selected_threshold_evaluation"] = _full_scene_evaluation(
            probability,
            probability_valid,
            ground_truth,
            args.threshold,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Diagnostic report written: {output_path}")
    if not report["eligible_for_model_selection"]:
        print(f"[INFO] role={args.role}: report is diagnostic-only and must not select a model")


if __name__ == "__main__":
    main()
