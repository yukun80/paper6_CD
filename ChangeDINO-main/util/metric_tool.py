"""
Copied and modified from
https://github.com/justchenhao/BIT_CD
"""

import numpy as np
from scipy import ndimage


###################       metrics      ###################
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

    def get_scores(self):
        scores_dict = cm2score(self.sum)
        return scores_dict

    def clear(self):
        self.initialized = False


###################      cm metrics      ###################
class ConfuseMatrixMeter(AverageMeter):
    """Computes and stores the average and current value"""

    def __init__(self, n_class):
        super(ConfuseMatrixMeter, self).__init__()
        self.n_class = n_class

    def update_cm(self, pr, gt, weight=1):
        val = get_confuse_matrix(num_classes=self.n_class, label_gts=gt, label_preds=pr)
        self.update(val, weight)
        current_score = cm2F1(val)
        return current_score

    def get_scores(self):
        scores_dict = cm2score(self.sum)
        return scores_dict


def harmonic_mean(xs):
    harmonic_mean = len(xs) / sum((x + 1e-6) ** -1 for x in xs)
    return harmonic_mean


def cm2F1(confusion_matrix):
    hist = confusion_matrix
    n_class = hist.shape[0]
    tp = np.diag(hist)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)
    # ---------------------------------------------------------------------- #
    # 1. Accuracy & Class Accuracy
    # ---------------------------------------------------------------------- #
    acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)

    # recall
    recall = tp / (sum_a1 + np.finfo(np.float32).eps)
    # acc_cls = np.nanmean(recall)

    # precision
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)

    # F1 score
    F1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
    mean_F1 = np.nanmean(F1)
    return mean_F1


def cm2score(confusion_matrix):
    hist = confusion_matrix
    n_class = hist.shape[0]
    tp = np.diag(hist)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)
    # ---------------------------------------------------------------------- #
    # 1. Accuracy & Class Accuracy
    # ---------------------------------------------------------------------- #
    acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)

    # recall
    recall = tp / (sum_a1 + np.finfo(np.float32).eps)
    # acc_cls = np.nanmean(recall)

    # precision
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)

    # F1 score
    F1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
    mean_F1 = np.nanmean(F1)
    # ---------------------------------------------------------------------- #
    # 2. Frequency weighted Accuracy & Mean IoU
    # ---------------------------------------------------------------------- #
    iu = tp / (sum_a1 + hist.sum(axis=0) - tp + np.finfo(np.float32).eps)
    mean_iu = np.nanmean(iu)

    freq = sum_a1 / (hist.sum() + np.finfo(np.float32).eps)
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    #
    cls_iou = dict(zip(["iou_" + str(i) for i in range(n_class)], iu))

    cls_precision = dict(
        zip(["precision_" + str(i) for i in range(n_class)], precision)
    )
    cls_recall = dict(zip(["recall_" + str(i) for i in range(n_class)], recall))
    cls_F1 = dict(zip(["F1_" + str(i) for i in range(n_class)], F1))

    score_dict = {"acc": acc, "miou": mean_iu, "mf1": mean_F1}
    score_dict.update(cls_iou)
    score_dict.update(cls_F1)
    score_dict.update(cls_precision)
    score_dict.update(cls_recall)
    return score_dict


def get_confuse_matrix(num_classes, label_gts, label_preds):
    def __fast_hist(label_gt, label_pred):
        """
        Collect values for Confusion Matrix
        For reference, please see: https://en.wikipedia.org/wiki/Confusion_matrix
        :param label_gt: <np.array> ground-truth
        :param label_pred: <np.array> prediction
        :return: <np.ndarray> values for confusion matrix
        """
        mask = (label_gt >= 0) & (label_gt < num_classes)
        hist = np.bincount(
            num_classes * label_gt[mask].astype(int) + label_pred[mask],
            minlength=num_classes**2,
        ).reshape(num_classes, num_classes)
        return hist

    confusion_matrix = np.zeros((num_classes, num_classes))
    for lt, lp in zip(label_gts, label_preds):
        confusion_matrix += __fast_hist(lt.flatten(), lp.flatten())
    return confusion_matrix


def get_mIoU(num_classes, label_gts, label_preds):
    confusion_matrix = get_confuse_matrix(num_classes, label_gts, label_preds)
    score_dict = cm2score(confusion_matrix)
    return score_dict["miou"]


def init_component_recall_stats():
    return {
        "tiny": {"hit": 0, "total": 0},
        "small": {"hit": 0, "total": 0},
        "large": {"hit": 0, "total": 0},
    }


def init_prediction_blob_stats():
    """统计大而紧致的误检连通域，用于约束整块伪斑。"""
    return {
        "compact_large_fp_count": 0,
        "compact_large_fp_area": 0,
        "pred_fg_area": 0,
    }


def update_component_recall_stats(
    stats,
    label_gt,
    label_pred,
    tiny_area_thresh: int = 100,
    small_area_thresh: int = 400,
):
    structure = np.ones((3, 3), dtype=np.int8)
    cc_map, n_components = ndimage.label(label_gt.astype(np.uint8) > 0, structure=structure)
    for component_id in range(1, n_components + 1):
        mask = cc_map == component_id
        area = int(mask.sum())
        if area <= 0:
            continue
        if area <= tiny_area_thresh:
            bucket = "tiny"
        elif area <= small_area_thresh:
            bucket = "small"
        else:
            bucket = "large"
        stats[bucket]["total"] += 1
        stats[bucket]["hit"] += int(np.any(label_pred[mask] > 0))
    return stats


def update_prediction_blob_stats(
    stats,
    label_gt,
    label_pred,
    area_thresh: int = 1024,
    fill_ratio_thresh: float = 0.45,
    max_gt_iou: float = 0.1,
):
    """累计预测中大而紧致的伪斑块。

    仅统计和 GT 几乎不重叠的实心大连通域，避免把细长水体或真实大洪水误判为伪块。
    """
    pred_mask = label_pred.astype(np.uint8) > 0
    gt_mask = label_gt.astype(np.uint8) > 0
    stats["pred_fg_area"] += int(pred_mask.sum())

    if not np.any(pred_mask):
        return stats

    structure = np.ones((3, 3), dtype=np.int8)
    cc_map, n_components = ndimage.label(pred_mask, structure=structure)
    for component_id in range(1, n_components + 1):
        mask = cc_map == component_id
        area = int(mask.sum())
        if area < area_thresh:
            continue

        ys, xs = np.where(mask)
        height = int(ys.max() - ys.min() + 1)
        width = int(xs.max() - xs.min() + 1)
        bbox_area = max(height * width, 1)
        fill_ratio = area / bbox_area
        if fill_ratio < fill_ratio_thresh:
            continue

        gt_overlap = int(np.logical_and(mask, gt_mask).sum())
        gt_support = gt_overlap / max(area, 1)
        if gt_support >= max_gt_iou:
            continue

        stats["compact_large_fp_count"] += 1
        stats["compact_large_fp_area"] += area
    return stats


def component_recall_scores(stats):
    scores = {}
    for bucket, values in stats.items():
        total = int(values["total"])
        hit = int(values["hit"])
        scores[f"{bucket}_gt_components"] = total
        scores[f"{bucket}_recall"] = hit / total if total > 0 else 0.0
    return scores


def prediction_blob_scores(stats):
    pred_fg_area = int(stats["pred_fg_area"])
    compact_large_fp_area = int(stats["compact_large_fp_area"])
    compact_large_fp_ratio = (
        compact_large_fp_area / pred_fg_area if pred_fg_area > 0 else 0.0
    )
    return {
        "pred_fg_area": pred_fg_area,
        "compact_large_fp_count": int(stats["compact_large_fp_count"]),
        "compact_large_fp_area": compact_large_fp_area,
        "compact_large_fp_ratio": compact_large_fp_ratio,
        "blob_precision": max(0.0, 1.0 - min(1.0, compact_large_fp_ratio)),
    }
