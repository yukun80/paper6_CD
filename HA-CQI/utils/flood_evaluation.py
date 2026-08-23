"""统一的二值洪水变化评估与阈值选择实现。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import ndimage


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
        for component_id in range(1, count + 1):
            mask = component_map == component_id
            area = int(np.count_nonzero(mask))
            if area <= 0:
                continue
            bucket = _component_bucket(area, self.tiny_area_thresh, self.small_area_thresh)
            coverage = float(np.count_nonzero(pred & mask)) / area
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


def false_positive_component_scores(
    target: np.ndarray,
    prediction: np.ndarray,
    valid_mask: np.ndarray,
) -> dict[str, float | int]:
    fp_mask = (
        np.asarray(prediction).astype(bool)
        & ~np.asarray(target).astype(bool)
        & np.asarray(valid_mask).astype(bool)
    )
    component_map, count = ndimage.label(fp_mask, structure=np.ones((3, 3), dtype=np.int8))
    areas = np.bincount(component_map.reshape(-1))[1:] if count else np.empty(0, dtype=np.int64)
    return {
        "fp_component_count": int(count),
        "largest_fp_component": int(areas.max()) if areas.size else 0,
        "p95_fp_component_area": float(np.percentile(areas, 95)) if areas.size else 0.0,
    }


class FloodEvaluationAccumulator:
    """内存常数级累计多阈值混淆矩阵、校准和参考阈值细粒度指标。"""

    def __init__(
        self,
        thresholds: Iterable[float],
        reference_threshold: float,
        calibration_bins: int = 15,
        tiny_area_thresh: int = 100,
        small_area_thresh: int = 400,
    ) -> None:
        values = sorted({float(value) for value in thresholds})
        if not values or any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError("thresholds must be a non-empty subset of [0, 1]")
        if not 0.0 <= reference_threshold <= 1.0:
            raise ValueError("reference_threshold must be within [0, 1]")
        if calibration_bins < 2:
            raise ValueError("calibration_bins must be >= 2")
        self.thresholds = np.asarray(values, dtype=np.float64)
        self.reference_threshold = float(reference_threshold)
        self.pos_pass_bins = np.zeros(len(values) + 1, dtype=np.int64)
        self.neg_pass_bins = np.zeros(len(values) + 1, dtype=np.int64)
        self.calibration_bins = int(calibration_bins)
        self.calibration_count = np.zeros(calibration_bins, dtype=np.int64)
        self.calibration_prob_sum = np.zeros(calibration_bins, dtype=np.float64)
        self.calibration_target_sum = np.zeros(calibration_bins, dtype=np.float64)
        self.brier_sum = 0.0
        self.valid_pixels = 0
        self.reference_confusion = np.zeros(4, dtype=np.int64)  # tp, fp, tn, fn
        self.component_stats = ComponentCoverageStats(tiny_area_thresh, small_area_thresh)
        self.background_tile_count = 0
        self.background_tile_fp_count = 0
        self.background_tile_fp_fractions: list[float] = []
        self.fp_component_areas: list[int] = []

    def update(
        self,
        probabilities: np.ndarray,
        target: np.ndarray,
        valid_mask: np.ndarray | None = None,
    ) -> None:
        probs = np.asarray(probabilities, dtype=np.float32)
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

        passed_count = np.searchsorted(self.thresholds, flat_probs, side="right")
        self.pos_pass_bins += np.bincount(
            passed_count[flat_gt], minlength=len(self.thresholds) + 1
        )
        self.neg_pass_bins += np.bincount(
            passed_count[~flat_gt], minlength=len(self.thresholds) + 1
        )

        calibration_index = np.minimum(
            (flat_probs * self.calibration_bins).astype(np.int64), self.calibration_bins - 1
        )
        self.calibration_count += np.bincount(
            calibration_index, minlength=self.calibration_bins
        )
        self.calibration_prob_sum += np.bincount(
            calibration_index, weights=flat_probs, minlength=self.calibration_bins
        )
        self.calibration_target_sum += np.bincount(
            calibration_index, weights=flat_gt.astype(np.float64), minlength=self.calibration_bins
        )
        self.brier_sum += float(np.square(flat_probs - flat_gt.astype(np.float64)).sum())
        self.valid_pixels += int(flat_probs.size)

        pred = probs >= self.reference_threshold
        reference = evaluate_binary_arrays(pred, gt, valid)
        self.reference_confusion += np.asarray(
            [reference["tp"], reference["fp"], reference["tn"], reference["fn"]],
            dtype=np.int64,
        )

        batch_probs = probs[None, ...] if probs.ndim == 2 else probs
        batch_gt = gt[None, ...] if gt.ndim == 2 else gt
        batch_valid = valid[None, ...] if valid.ndim == 2 else valid
        for prob_item, gt_item, valid_item in zip(batch_probs, batch_gt, batch_valid):
            pred_item = prob_item >= self.reference_threshold
            self.component_stats.update(gt_item, pred_item, valid_item)
            valid_count = int(np.count_nonzero(valid_item))
            if valid_count > 0 and not np.any(gt_item & valid_item):
                fp_fraction = float(np.count_nonzero(pred_item & valid_item)) / valid_count
                self.background_tile_count += 1
                self.background_tile_fp_count += int(fp_fraction > 0.0)
                self.background_tile_fp_fractions.append(fp_fraction)
            fp_components = false_positive_component_scores(gt_item, pred_item, valid_item)
            if int(fp_components["fp_component_count"]) > 0:
                fp_mask = pred_item & ~gt_item & valid_item
                component_map, count = ndimage.label(
                    fp_mask, structure=np.ones((3, 3), dtype=np.int8)
                )
                self.fp_component_areas.extend(
                    int(value) for value in np.bincount(component_map.reshape(-1))[1 : count + 1]
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

    def _pr_auc(self, threshold_metrics: list[dict[str, float | int]]) -> float:
        points = [(0.0, 1.0)]
        points.extend(
            (float(item["recall_1"]), float(item["precision_1"]))
            for item in reversed(threshold_metrics)
        )
        positive_total = int(self.pos_pass_bins.sum())
        prevalence = positive_total / max(self.valid_pixels, 1)
        points.append((1.0, prevalence))
        points = sorted(points, key=lambda item: item[0])
        recall = np.asarray([item[0] for item in points], dtype=np.float64)
        precision = np.asarray([item[1] for item in points], dtype=np.float64)
        return float(np.trapz(precision, recall))

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

        nonempty = self.calibration_count > 0
        mean_prob = np.zeros_like(self.calibration_prob_sum)
        mean_target = np.zeros_like(self.calibration_target_sum)
        mean_prob[nonempty] = self.calibration_prob_sum[nonempty] / self.calibration_count[nonempty]
        mean_target[nonempty] = (
            self.calibration_target_sum[nonempty] / self.calibration_count[nonempty]
        )
        ece = float(
            np.sum(
                self.calibration_count[nonempty]
                / max(self.valid_pixels, 1)
                * np.abs(mean_prob[nonempty] - mean_target[nonempty])
            )
        )
        return {
            "selected": selected,
            "reference": reference,
            "thresholds": threshold_results,
            "calibration": {
                "pr_auc": self._pr_auc(threshold_results),
                "brier": self.brier_sum / max(self.valid_pixels, 1),
                "ece": ece,
                "ece_bins": self.calibration_bins,
            },
        }
