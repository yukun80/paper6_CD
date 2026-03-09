import os
from typing import Any, Dict, Optional, Tuple

import torch


def save_checkpoint(
    output_dir: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    best_metric: float,
    metric_name: str,
    is_best: bool,
) -> Tuple[str, Optional[str]]:
    """保存 latest 与 best checkpoint。"""
    os.makedirs(output_dir, exist_ok=True)
    state = {
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_metric": float(best_metric),
        "metric_name": metric_name,
    }
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()

    latest_path = os.path.join(output_dir, "latest.pth")
    torch.save(state, latest_path)

    best_path = None
    if is_best:
        best_path = os.path.join(output_dir, "best.pth")
        torch.save(state, best_path)

    return latest_path, best_path


def load_checkpoint(
    ckpt_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    """加载 checkpoint；支持仅加载模型参数。"""
    ckpt = torch.load(ckpt_path, map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=False)

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])

    return {
        "epoch": int(ckpt.get("epoch", -1)),
        "best_metric": float(ckpt.get("best_metric", 0.0)),
        "metric_name": ckpt.get("metric_name", "mIoU"),
    }
