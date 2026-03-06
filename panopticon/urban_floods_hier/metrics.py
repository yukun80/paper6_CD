from typing import Dict, Sequence

import torch


def update_confusion_matrix(
    conf: torch.Tensor,
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> torch.Tensor:
    """累积混淆矩阵（忽略 ignore_index）。"""
    valid = target != ignore_index
    if valid.sum() == 0:
        return conf
    pred = pred[valid].view(-1)
    target = target[valid].view(-1)
    idx = target * num_classes + pred
    binc = torch.bincount(idx, minlength=num_classes * num_classes)
    conf += binc.view(num_classes, num_classes)
    return conf


def summarize_from_confusion(
    conf: torch.Tensor,
    eps: float = 1e-6,
    positive_classes: Sequence[int] = (1, 2),
) -> Dict[str, float]:
    """根据混淆矩阵计算总体与分类型指标。"""
    conf = conf.float()
    tp = conf.diag()
    fp = conf.sum(dim=0) - tp
    fn = conf.sum(dim=1) - tp

    iou = tp / (tp + fp + fn + eps)
    f1 = 2 * tp / (2 * tp + fp + fn + eps)
    acc = tp.sum() / (conf.sum() + eps)

    metrics = {
        "mIoU": float(iou.mean().item()),
        "mF1": float(f1.mean().item()),
        "pixel_acc": float(acc.item()),
    }
    for i in range(iou.numel()):
        metrics[f"IoU_{i}"] = float(iou[i].item())
        metrics[f"F1_{i}"] = float(f1[i].item())

    pos_idx = [int(i) for i in positive_classes if 0 <= int(i) < iou.numel()]
    if pos_idx:
        metrics["pos_mIoU"] = float(iou[pos_idx].mean().item())
        metrics["pos_mF1"] = float(f1[pos_idx].mean().item())
    else:
        metrics["pos_mIoU"] = float(iou.mean().item())
        metrics["pos_mF1"] = float(f1.mean().item())
    return metrics
