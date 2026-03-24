"""评估接口预留。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import rasterio


def evaluate_binary_mask(pred_mask: np.ndarray, gt_path: Optional[str], valid_mask: np.ndarray) -> Optional[Dict[str, object]]:
    if not gt_path:
        return None
    gt_file = Path(gt_path)
    if not gt_file.exists():
        return None

    with rasterio.open(gt_file) as ds:
        gt = ds.read(1)
        gt_nodata = ds.nodata

    gt_valid = valid_mask.copy()
    if gt_nodata is not None:
        gt_valid &= gt != gt_nodata

    pred = pred_mask[gt_valid].astype(bool)
    truth = gt[gt_valid].astype(bool)
    tp = int(np.logical_and(pred, truth).sum())
    tn = int(np.logical_and(~pred, ~truth).sum())
    fp = int(np.logical_and(pred, ~truth).sum())
    fn = int(np.logical_and(~pred, truth).sum())
    total = tp + tn + fp + fn
    if total == 0:
        return None

    oa = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    p0 = oa
    pe = (((tp + fp) * (tp + fn)) + ((fn + tn) * (fp + tn))) / (total * total)
    kappa = (p0 - pe) / (1 - pe) if (1 - pe) > 0 else 0.0
    return {
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "OA": oa,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Kappa": kappa,
    }
