#!/usr/bin/env python3
"""
AWCA-Net 评估入口。

功能:
1. 加载训练得到的 checkpoint（best/last）。
2. 在指定 split（Val/Test）上计算二分类变化检测指标。
3. 输出 JSON 报告，便于复现实验记录。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader

from dataset.Transforms import Compose, Normalize, Scale, ToTensor
from dataset.dataset import Dataset
from loss.BCEDiceLoss import BCEDiceLoss

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


@dataclass
class BinaryMetrics:
    precision: float
    recall: float
    f1: float
    iou: float
    oa: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AWCA-Net checkpoint.")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "datasets/VarFloods/tiles")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--split", type=str, default="Test", choices=["Val", "Test", "Train"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=None, help="Optional output JSON path.")
    return parser.parse_args()


def compute_metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> BinaryMetrics:
    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    oa = (tp + tn) / (tp + fp + fn + tn + eps)
    return BinaryMetrics(precision=precision, recall=recall, f1=f1, iou=iou, oa=oa)


def load_norm_from_ckpt(ckpt: Dict) -> Tuple[list[float], list[float]]:
    mean = ckpt.get("mean")
    std = ckpt.get("std")
    if isinstance(mean, list) and isinstance(std, list) and len(mean) == 6 and len(std) == 6:
        return [float(x) for x in mean], [float(x) for x in std]
    return [0.5] * 6, [0.5] * 6


def main() -> None:
    args = parse_args()
    try:
        from AWCANet import AWCANet
    except Exception as exc:
        raise RuntimeError(
            "Failed to import AWCANet. Please ensure dependencies are installed: "
            "torch, torchvision, timm, einops, dropblock."
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(str(args.ckpt), map_location="cpu")
    mean, std = load_norm_from_ckpt(ckpt)

    tf = Compose(
        [
            Scale(args.tile_size, args.tile_size),
            Normalize(mean, std),
            ToTensor(),
        ]
    )
    ds = Dataset(file_root=str(args.data_root), mode=args.split, transform=tf)
    if len(ds) == 0:
        raise RuntimeError(f"Split {args.split} is empty under {args.data_root}")

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = AWCANet(pvt_path=None).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    total_loss = 0.0
    n_batches = 0
    tp = fp = fn = tn = 0

    with torch.no_grad():
        for image, label, _ in loader:
            image = image.to(device, non_blocking=True).float()
            label = label.to(device, non_blocking=True).float()
            outputs = model(image[:, :3, :, :], image[:, 3:, :, :])

            loss = 0.0
            for out in outputs:
                if out.shape[-2:] != label.shape[-2:]:
                    out = torch.nn.functional.interpolate(
                        out,
                        size=label.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                loss = loss + BCEDiceLoss(out, label)
            loss = loss / len(outputs)
            total_loss += float(loss.item())
            n_batches += 1

            pred_logits = outputs[-1]
            if pred_logits.shape[-2:] != label.shape[-2:]:
                pred_logits = torch.nn.functional.interpolate(
                    pred_logits,
                    size=label.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            pred_prob = torch.sigmoid(pred_logits)
            pred_bin = pred_prob >= args.threshold
            gt_bin = label >= 0.5
            tp += int(torch.logical_and(pred_bin, gt_bin).sum().item())
            fp += int(torch.logical_and(pred_bin, torch.logical_not(gt_bin)).sum().item())
            fn += int(torch.logical_and(torch.logical_not(pred_bin), gt_bin).sum().item())
            tn += int(torch.logical_and(torch.logical_not(pred_bin), torch.logical_not(gt_bin)).sum().item())

    metrics = compute_metrics_from_counts(tp=tp, fp=fp, fn=fn, tn=tn)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "data_root": str(args.data_root),
        "ckpt": str(args.ckpt),
        "samples": len(ds),
        "batch_size": args.batch_size,
        "threshold": args.threshold,
        "loss": total_loss / max(n_batches, 1),
        "metrics": asdict(metrics),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    out_path = args.out
    if out_path is None:
        out_path = args.ckpt.parent.parent / f"eval_{args.split.lower()}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[eval_awca] Saved report to: {out_path}")


if __name__ == "__main__":
    main()
