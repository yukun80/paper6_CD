"""统一的二值洪水变化评估与阈值选择实现。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import ndimage

from .prediction import threshold_probability


EPS = np.finfo(np.float64).eps


def build_threshold_grid(start: float, stop: float, step: float) -> list[float]:
    """用整数步生成稳定阈值网格，避免浮点累计误差。"""
    if not 0.0 <= start <= stop <= 1.0:
        raise ValueError("threshold range must satisfy 0 <= start <= stop <= 1")
    if step <= 0.0:
        raise ValueError("threshold step must be positive")
    count = int(math.floor((stop - start) / step + 1e-9))
    values = [round(start + index * step, 10) for index in range(count + 1)]
    if not values or values[-1] < stop - 1e-9:
        values.append(round(stop, 10))
    return values


def confusion_metrics(tp: int, fp: int, tn: int, fn: int) -> dict[str, float | int]:
    """按 Flood=1 口径生成像素级指标。"""
    iou = tp / (tp + fp + fn + EPS)
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    f1 = 2.0 * tp / (2.0 * tp + fp + fn + EPS)
    valid = tp + fp + tn + fn
    return {
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "valid_pixels": int(valid),
        "iou_1": float(iou),
        "F1_1": float(f1),
        "precision_1": float(precision),
        "recall_1": float(recall),
        "acc": float((tp + tn) / (valid + EPS)),
    }


def evaluate_binary_arrays(
    prediction: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> dict[str, float | int]:
    pred = np.asarray(prediction).astype(bool, copy=False)
    gt = np.asarray(target).astype(bool, copy=False)
    if pred.shape != gt.shape:
        raise ValueError(f"prediction/target shape mismatch: {pred.shape} != {gt.shape}")
    valid = np.ones(gt.shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask).astype(bool)
    if valid.shape != gt.shape:
        raise ValueError(f"valid_mask shape mismatch: {valid.shape} != {gt.shape}")
    tp = int(np.count_nonzero(valid & pred & gt))
    fp = int(np.count_nonzero(valid & pred & ~gt))
    tn = int(np.count_nonzero(valid & ~pred & ~gt))
    fn = int(np.count_nonzero(valid & ~pred & gt))
    result = confusion_metrics(tp, fp, tn, fn)
    result["ignored_pixels"] = int(gt.size - np.count_nonzero(valid))
    return result


def select_primary_threshold(
    candidates: Iterable[dict[str, float | int]],
) -> dict[str, float | int]:
    """按 IoU、Precision、较高阈值依次打破并列。"""
    items = list(candidates)
    if not items:
        raise ValueError("threshold candidates must not be empty")
    return max(
        items,
        key=lambda item: (
            float(item["iou_1"]),
            float(item["precision_1"]),
            float(item["threshold"]),
        ),
    )


def _component_bucket(area: int, tiny_area_thresh: int, small_area_thresh: int) -> str:
    if area <= tiny_area_thresh:
        return "tiny"
    if area <= small_area_thresh:
        return "small"
    return "large"


@dataclass
class ComponentCoverageStats:
    tiny_area_thresh: int = 100
    small_area_thresh: int = 400

    def __post_init__(self) -> None:
        self.total = {bucket: 0 for bucket in ("tiny", "small", "large")}
        self.hit10 = {bucket: 0 for bucket in ("tiny", "small", "large")}
        self.hit25 = {bucket: 0 for bucket in ("tiny", "small", "large")}

    def update(self, target: np.ndarray, prediction: np.ndarray, valid_mask: np.ndarray) -> None:
        structure = np.ones((3, 3), dtype=np.int8)
        gt = np.asarray(target).astype(bool) & np.asarray(valid_mask).astype(bool)
        pred = np.asarray(prediction).astype(bool) & np.asarray(valid_mask).astype(bool)
        component_map, count = ndimage.label(gt, structure=structure)
        # 按组件一次汇总面积与命中数，避免为每个组件扫描整张图。
        areas = np.bincount(component_map.reshape(-1), minlength=count + 1)
        hits = np.bincount(component_map[pred], minlength=count + 1)
        for component_id in range(1, count + 1):
            area = int(areas[component_id])
            if area <= 0:
                continue
            bucket = _component_bucket(area, self.tiny_area_thresh, self.small_area_thresh)
            coverage = float(hits[component_id]) / area
            self.total[bucket] += 1
            self.hit10[bucket] += int(coverage >= 0.10)
            self.hit25[bucket] += int(coverage >= 0.25)

    def scores(self) -> dict[str, float | int]:
        result: dict[str, float | int] = {}
        for bucket in ("tiny", "small", "large"):
            total = int(self.total[bucket])
            result[f"{bucket}_gt_components"] = total
            result[f"{bucket}_recall_cov10"] = self.hit10[bucket] / total if total else 0.0
            result[f"{bucket}_recall_cov25"] = self.hit25[bucket] / total if total else 0.0
        return result


def _false_positive_component_areas(
    target: np.ndarray,
    prediction: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """返回按连通标记顺序排列的误报面积，供单图与累计统计共用。"""
    fp_mask = (
        np.asarray(prediction).astype(bool)
        & ~np.asarray(target).astype(bool)
        & np.asarray(valid_mask).astype(bool)
    )
    component_map, count = ndimage.label(fp_mask, structure=np.ones((3, 3), dtype=np.int8))
    return np.bincount(component_map.reshape(-1))[1:] if count else np.empty(0, dtype=np.int64)


def false_positive_component_scores(
    target: np.ndarray,
    prediction: np.ndarray,
    valid_mask: np.ndarray,
) -> dict[str, float | int]:
    areas = _false_positive_component_areas(target, prediction, valid_mask)
    return {
        "fp_component_count": int(areas.size),
        "largest_fp_component": int(areas.max()) if areas.size else 0,
        "p95_fp_component_area": float(np.percentile(areas, 95)) if areas.size else 0.0,
    }


def boundary_match_counts(
    target: np.ndarray,
    prediction: np.ndarray,
    valid_mask: np.ndarray,
    tolerance: int,
) -> dict[str, int]:
    """返回容差匹配的洪水内边界计数，忽略 nodata 邻域。"""
    if tolerance < 0:
        raise ValueError("boundary tolerance must be non-negative")
    structure = np.ones((3, 3), dtype=bool)
    valid = np.asarray(valid_mask).astype(bool)
    gt = np.asarray(target).astype(bool) & valid
    pred = np.asarray(prediction).astype(bool) & valid
    interior_valid = ndimage.binary_erosion(valid, structure=structure, border_value=0)
    gt_boundary = gt & ~ndimage.binary_erosion(gt, structure=structure, border_value=0)
    pred_boundary = pred & ~ndimage.binary_erosion(pred, structure=structure, border_value=0)
    gt_boundary &= interior_valid
    pred_boundary &= interior_valid

    if tolerance > 0:
        match_structure = ndimage.generate_binary_structure(2, 2)
        gt_tolerance = ndimage.binary_dilation(
            gt_boundary,
            structure=match_structure,
            iterations=tolerance,
        )
        pred_tolerance = ndimage.binary_dilation(
            pred_boundary,
            structure=match_structure,
            iterations=tolerance,
        )
    else:
        gt_tolerance = gt_boundary
        pred_tolerance = pred_boundary
    return {
        "matched_prediction": int(np.count_nonzero(pred_boundary & gt_tolerance)),
        "prediction_boundary": int(np.count_nonzero(pred_boundary)),
        "matched_target": int(np.count_nonzero(gt_boundary & pred_tolerance)),
        "target_boundary": int(np.count_nonzero(gt_boundary)),
    }


def boundary_f1_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    valid_mask: np.ndarray | None = None,
    tolerance: int = 2,
) -> dict[str, float | int]:
    """计算容差边界 Precision/Recall/F1，双空边界定义为 1。"""
    gt = np.asarray(target)
    valid = np.ones(gt.shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask)
    counts = boundary_match_counts(gt, prediction, valid, tolerance)
    return {**counts, **_boundary_metrics_from_counts(counts, tolerance)}


def _boundary_metrics_from_counts(
    counts: dict[str, int], tolerance: int,
) -> dict[str, float]:
    """统一边界计数到指标的换算，保留空边界约定和原浮点运算顺序。"""
    pred_total = counts["prediction_boundary"]
    target_total = counts["target_boundary"]
    precision = (
        counts["matched_prediction"] / pred_total
        if pred_total
        else float(target_total == 0)
    )
    recall = (
        counts["matched_target"] / target_total
        if target_total
        else float(pred_total == 0)
    )
    f1 = 2.0 * precision * recall / (precision + recall + EPS)
    return {
        f"boundary_precision_tol{tolerance}": float(precision),
        f"boundary_recall_tol{tolerance}": float(recall),
        f"boundary_f1_tol{tolerance}": float(f1),
    }


class FloodEvaluationAccumulator:
    """内存常数级累计多阈值混淆矩阵和参考阈值细粒度指标。"""

    def __init__(
        self,
        thresholds: Iterable[float],
        reference_threshold: float,
        tiny_area_thresh: int = 100,
        small_area_thresh: int = 400,
    ) -> None:
        values = sorted({float(value) for value in thresholds})
        if not values or any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError("thresholds must be a non-empty subset of [0, 1]")
        if not 0.0 <= reference_threshold <= 1.0:
            raise ValueError("reference_threshold must be within [0, 1]")
        self.thresholds = np.asarray(values, dtype=np.float64)
        # 比较与报告均保留 FP64，避免邻近阈值被 FP32 舍入合并。
        self._comparison_thresholds = self.thresholds.copy()
        self.reference_threshold = float(reference_threshold)
        self.pos_pass_bins = np.zeros(len(values) + 1, dtype=np.int64)
        self.neg_pass_bins = np.zeros(len(values) + 1, dtype=np.int64)
        self.valid_pixels = 0
        self.reference_confusion = np.zeros(4, dtype=np.int64)  # tp, fp, tn, fn
        self.component_stats = ComponentCoverageStats(tiny_area_thresh, small_area_thresh)
        self.background_tile_count = 0
        self.background_tile_fp_count = 0
        self.background_tile_fp_fractions: list[float] = []
        self.fp_component_areas: list[int] = []
        self.boundary_tolerances = (2, 4)
        self.boundary_counts = {
            tolerance: {
                "matched_prediction": 0,
                "prediction_boundary": 0,
                "matched_target": 0,
                "target_boundary": 0,
            }
            for tolerance in self.boundary_tolerances
        }

    def update(
        self,
        probabilities: np.ndarray,
        target: np.ndarray,
        valid_mask: np.ndarray | None = None,
    ) -> None:
        probs = np.asarray(probabilities, dtype=np.float64)
        gt = np.asarray(target).astype(bool)
        if probs.shape != gt.shape:
            raise ValueError(f"probabilities/target shape mismatch: {probs.shape} != {gt.shape}")
        valid = np.ones(gt.shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask).astype(bool)
        if valid.shape != gt.shape:
            raise ValueError(f"valid_mask shape mismatch: {valid.shape} != {gt.shape}")
        valid &= np.isfinite(probs)
        flat_probs = np.clip(probs[valid].astype(np.float64, copy=False), 0.0, 1.0)
        flat_gt = gt[valid]
        if flat_probs.size == 0:
            return

        passed_count = np.searchsorted(
            self._comparison_thresholds, flat_probs, side="right"
        )
        self.pos_pass_bins += np.bincount(
            passed_count[flat_gt], minlength=len(self.thresholds) + 1
        )
        self.neg_pass_bins += np.bincount(
            passed_count[~flat_gt], minlength=len(self.thresholds) + 1
        )

        self.valid_pixels += int(flat_probs.size)

        pred = threshold_probability(probs, self.reference_threshold)
        reference = evaluate_binary_arrays(pred, gt, valid)
        self.reference_confusion += np.asarray(
            [reference["tp"], reference["fp"], reference["tn"], reference["fn"]],
            dtype=np.int64,
        )

        batch_predictions = pred[None, ...] if pred.ndim == 2 else pred
        batch_gt = gt[None, ...] if gt.ndim == 2 else gt
        batch_valid = valid[None, ...] if valid.ndim == 2 else valid
        for pred_item, gt_item, valid_item in zip(batch_predictions, batch_gt, batch_valid):
            self.component_stats.update(gt_item, pred_item, valid_item)
            for tolerance in self.boundary_tolerances:
                counts = boundary_match_counts(
                    gt_item,
                    pred_item,
                    valid_item,
                    tolerance,
                )
                for key, value in counts.items():
                    self.boundary_counts[tolerance][key] += int(value)
            valid_count = int(np.count_nonzero(valid_item))
            if valid_count > 0 and not np.any(gt_item & valid_item):
                fp_fraction = float(np.count_nonzero(pred_item & valid_item)) / valid_count
                self.background_tile_count += 1
                self.background_tile_fp_count += int(fp_fraction > 0.0)
                self.background_tile_fp_fractions.append(fp_fraction)
            self.fp_component_areas.extend(
                int(value)
                for value in _false_positive_component_areas(gt_item, pred_item, valid_item)
            )

    def threshold_metrics(self) -> list[dict[str, float | int]]:
        pos_suffix = np.cumsum(self.pos_pass_bins[::-1])[::-1]
        neg_suffix = np.cumsum(self.neg_pass_bins[::-1])[::-1]
        positive_total = int(self.pos_pass_bins.sum())
        negative_total = int(self.neg_pass_bins.sum())
        results: list[dict[str, float | int]] = []
        for index, threshold in enumerate(self.thresholds):
            tp = int(pos_suffix[index + 1])
            fp = int(neg_suffix[index + 1])
            fn = positive_total - tp
            tn = negative_total - fp
            metrics = confusion_metrics(tp, fp, tn, fn)
            metrics["threshold"] = float(threshold)
            results.append(metrics)
        return results

    def scores(self) -> dict[str, object]:
        threshold_results = self.threshold_metrics()
        selected = select_primary_threshold(threshold_results)
        tp, fp, tn, fn = (int(value) for value in self.reference_confusion)
        reference = confusion_metrics(tp, fp, tn, fn)
        reference["threshold"] = self.reference_threshold
        reference.update(self.component_stats.scores())
        background_fractions = np.asarray(self.background_tile_fp_fractions, dtype=np.float64)
        reference.update(
            {
                "background_tile_count": int(self.background_tile_count),
                "background_tile_fp_count": int(self.background_tile_fp_count),
                "background_tile_fp_rate": (
                    self.background_tile_fp_count / self.background_tile_count
                    if self.background_tile_count
                    else 0.0
                ),
                "background_tile_mean_fp_fraction": (
                    float(background_fractions.mean()) if background_fractions.size else 0.0
                ),
                "background_tile_p95_fp_fraction": (
                    float(np.percentile(background_fractions, 95))
                    if background_fractions.size
                    else 0.0
                ),
            }
        )
        for tolerance, counts in self.boundary_counts.items():
            reference.update(_boundary_metrics_from_counts(counts, tolerance))
        fp_areas = np.asarray(self.fp_component_areas, dtype=np.int64)
        reference.update(
            {
                "fp_component_count": int(fp_areas.size),
                "largest_fp_component": int(fp_areas.max()) if fp_areas.size else 0,
                "p95_fp_component_area": (
                    float(np.percentile(fp_areas, 95)) if fp_areas.size else 0.0
                ),
                "fp_valid_ratio": fp / max(self.valid_pixels, 1),
            }
        )

        return {
            "selected": selected,
            "reference": reference,
            "thresholds": threshold_results,
        }
