from typing import Dict, List

import torch


def update_confusion_matrix(
    conf: torch.Tensor,
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int,
) -> torch.Tensor:
    valid = target != ignore_index
    if valid.sum() == 0:
        return conf
    p = pred[valid].view(-1).long()
    t = target[valid].view(-1).long()
    idx = t * num_classes + p
    bins = torch.bincount(idx, minlength=num_classes * num_classes)
    return conf + bins.view(num_classes, num_classes)


def _safe_div(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    return num / den.clamp(min=1e-6)


def summarize_metrics(conf: torch.Tensor, class_names: List[str]) -> Dict[str, float]:
    tp = conf.diag().float()
    fp = conf.sum(dim=0).float() - tp
    fn = conf.sum(dim=1).float() - tp

    iou = _safe_div(tp, tp + fp + fn)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    out: Dict[str, float] = {
        "mIoU": float(iou.mean().item()),
        "mF1": float(f1.mean().item()),
        "OA": float(_safe_div(tp.sum(), conf.sum().float()).item()),
    }

    for i, name in enumerate(class_names):
        out[f"IoU_{name}"] = float(iou[i].item())
        out[f"F1_{name}"] = float(f1[i].item())
        out[f"Precision_{name}"] = float(precision[i].item())
        out[f"Recall_{name}"] = float(recall[i].item())

    # 特别关注 open/urban 的正类表现
    if len(class_names) >= 3:
        out["open_IoU"] = out.get("IoU_flood-open", out.get("IoU_1", 0.0))
        out["open_F1"] = out.get("F1_flood-open", out.get("F1_1", 0.0))
        out["urban_IoU"] = out.get("IoU_flood-urban", out.get("IoU_2", 0.0))
        out["urban_F1"] = out.get("F1_flood-urban", out.get("F1_2", 0.0))

    return out
