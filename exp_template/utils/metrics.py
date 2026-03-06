"""语义分割指标工具。"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch


class SegMetricMeter:
    """语义分割混淆矩阵指标累加器。"""

    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.confusion = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64)

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred = pred.detach().cpu().long()
        target = target.detach().cpu().long()
        mask = (target != self.ignore_index) & (target >= 0) & (target < self.num_classes)
        if not torch.any(mask):
            return
        pred = pred[mask]
        target = target[mask]
        index = target * self.num_classes + pred
        binc = torch.bincount(index, minlength=self.num_classes**2)
        self.confusion += binc.view(self.num_classes, self.num_classes)

    def compute(self, pos_classes: Sequence[int] = (1, 2)) -> Dict[str, float]:
        return compute_metrics_from_confusion(self.confusion, pos_classes=pos_classes)


def compute_metrics_from_confusion(confusion: torch.Tensor, pos_classes: Sequence[int] = (1, 2)) -> Dict[str, float]:
    """由混淆矩阵计算 IoU/Precision/Recall/F1/OA 等指标。"""

    conf = confusion.double()
    eps = 1e-12
    num_classes = conf.shape[0]

    per_iou = []
    per_precision = []
    per_recall = []
    per_f1 = []
    metrics: Dict[str, float] = {}

    for c in range(num_classes):
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = conf[c, :].sum() - tp

        precision = (tp / (tp + fp + eps)).item()
        recall = (tp / (tp + fn + eps)).item()
        iou = (tp / (tp + fp + fn + eps)).item()
        f1 = (2 * tp / (2 * tp + fp + fn + eps)).item()
        metrics[f"Precision_{c}"] = precision
        metrics[f"Recall_{c}"] = recall
        metrics[f"IoU_{c}"] = iou
        metrics[f"F1_{c}"] = f1
        per_precision.append(precision)
        per_recall.append(recall)
        per_iou.append(iou)
        per_f1.append(f1)

    metrics["mPrecision"] = float(np.mean(per_precision))
    metrics["mRecall"] = float(np.mean(per_recall))
    metrics["mIoU"] = float(np.mean(per_iou))
    metrics["mF1"] = float(np.mean(per_f1))

    if len(pos_classes) > 0:
        valid_pos_classes = [i for i in pos_classes if 0 <= i < num_classes]
        if len(valid_pos_classes) == 0:
            valid_pos_classes = list(range(num_classes))
        pos_precision = [per_precision[i] for i in valid_pos_classes]
        pos_recall = [per_recall[i] for i in valid_pos_classes]
        pos_iou = [per_iou[i] for i in valid_pos_classes]
        pos_f1 = [per_f1[i] for i in valid_pos_classes]
        metrics["pos_mPrecision"] = float(np.mean(pos_precision))
        metrics["pos_mRecall"] = float(np.mean(pos_recall))
        metrics["pos_mIoU"] = float(np.mean(pos_iou))
        metrics["pos_mF1"] = float(np.mean(pos_f1))
    else:
        metrics["pos_mPrecision"] = metrics["mPrecision"]
        metrics["pos_mRecall"] = metrics["mRecall"]
        metrics["pos_mIoU"] = metrics["mIoU"]
        metrics["pos_mF1"] = metrics["mF1"]

    total = conf.sum().item()
    correct = torch.diag(conf).sum().item()
    metrics["overall_acc"] = float(correct / max(total, eps))
    metrics["OA"] = metrics["overall_acc"]
    return metrics
