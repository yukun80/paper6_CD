#!/usr/bin/env python3
"""
AWCA-Net 训练入口（面向 VarFloods 预处理后的 tiles）。

特性:
1. 自动读取数据集推荐归一化统计（recommended_norm.json）。
2. 支持自动下载 PVTv2-B2 预训练权重。
3. 单卡训练，支持 AMP 混合精度。
4. 每个 epoch 在 Val 上评估，按 Val F1 保存最佳权重。
"""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from dataset.Transforms import Compose, Normalize, RandomExchange, RandomFlip, Scale, ToTensor
from dataset.dataset import Dataset
from loss.BCEDiceLoss import BCEDiceLoss

"""
python AWCA-Net-main/train_awca.py \
    --data-root datasets/VarFloods/tiles \
    --work-dir AWCA-Net-main/work_dirs/varfloods_pro_awca \
    --epochs 200 \
    --batch-size 12 \
    --num-workers 8 \
    --amp
"""


PVT_B2_URL = "https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth"
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
    parser = argparse.ArgumentParser(description="Train AWCA-Net on tiled VarFloods dataset.")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "datasets/VarFloods/tiles")
    parser.add_argument("--work-dir", type=Path, default=SCRIPT_DIR / "work_dirs/varfloods_pro_awca")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for validation metrics.")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--pvt-weights", type=Path, default=SCRIPT_DIR / "weights/pvt_v2_b2.pth")
    parser.add_argument("--disable-pretrained", action="store_true", help="Do not load PVT pretrained weights.")
    parser.add_argument("--resume", type=Path, default=None, help="Resume from a checkpoint path.")
    parser.add_argument("--amp", dest="amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_pvt_weights(weight_path: Path) -> Path:
    """
    自动下载 PVTv2-B2 预训练权重。
    若下载失败，直接抛出异常，避免无感退化成随机初始化。
    """
    if weight_path.exists():
        return weight_path
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        print(f"[train_awca] Downloading pretrained weight: {PVT_B2_URL}")
        urllib.request.urlretrieve(PVT_B2_URL, str(weight_path))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download PVT weights from {PVT_B2_URL}. "
            f"Please place weights manually at {weight_path}"
        ) from exc
    if not weight_path.exists():
        raise RuntimeError(f"Download reported success but file missing: {weight_path}")
    return weight_path


def load_mean_std(data_root: Path) -> Tuple[list[float], list[float]]:
    norm_path = data_root / "recommended_norm.json"
    if not norm_path.exists():
        print(f"[train_awca] {norm_path} not found, fallback to default mean/std=0.5.")
        return [0.5] * 6, [0.5] * 6
    content = json.loads(norm_path.read_text(encoding="utf-8"))
    fields = content.get("recommended_config_fields", {})
    mean = fields.get("mean")
    std = fields.get("std")
    if not (isinstance(mean, list) and isinstance(std, list) and len(mean) == 6 and len(std) == 6):
        raise ValueError(f"Invalid mean/std format in {norm_path}")
    return [float(x) for x in mean], [float(x) for x in std]


def build_dataloaders(
    data_root: Path,
    tile_size: int,
    mean: list[float],
    std: list[float],
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    train_tf = Compose(
        [
            Scale(tile_size, tile_size),
            RandomFlip(),
            RandomExchange(),
            Normalize(mean, std),
            ToTensor(),
        ]
    )
    val_tf = Compose(
        [
            Scale(tile_size, tile_size),
            Normalize(mean, std),
            ToTensor(),
        ]
    )

    train_ds = Dataset(file_root=str(data_root), mode="Train", transform=train_tf)
    val_ds = Dataset(file_root=str(data_root), mode="Val", transform=val_tf)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(f"Empty split found in {data_root}; got Train={len(train_ds)}, Val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


def compute_metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> BinaryMetrics:
    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    oa = (tp + tn) / (tp + fp + fn + tn + eps)
    return BinaryMetrics(precision=precision, recall=recall, f1=f1, iou=iou, oa=oa)


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> Tuple[float, BinaryMetrics]:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    tp = fp = fn = tn = 0
    with torch.no_grad():
        for image, label, _ in loader:
            image = image.to(device, non_blocking=True).float()
            label = label.to(device, non_blocking=True).float()
            outputs = model(image[:, :3, :, :], image[:, 3:, :, :])

            batch_loss = 0.0
            for out in outputs:
                if out.shape[-2:] != label.shape[-2:]:
                    out = torch.nn.functional.interpolate(
                        out,
                        size=label.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                batch_loss = batch_loss + BCEDiceLoss(out, label)
            batch_loss = batch_loss / len(outputs)
            total_loss += float(batch_loss.item())
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
            pred_bin = pred_prob >= threshold
            gt_bin = label >= 0.5

            tp += int(torch.logical_and(pred_bin, gt_bin).sum().item())
            fp += int(torch.logical_and(pred_bin, torch.logical_not(gt_bin)).sum().item())
            fn += int(torch.logical_and(torch.logical_not(pred_bin), gt_bin).sum().item())
            tn += int(torch.logical_and(torch.logical_not(pred_bin), torch.logical_not(gt_bin)).sum().item())

    avg_loss = total_loss / max(n_batches, 1)
    metrics = compute_metrics_from_counts(tp=tp, fp=fp, fn=fn, tn=tn)
    return avg_loss, metrics


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for image, label, _ in loader:
        image = image.to(device, non_blocking=True).float()
        label = label.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
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

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(n_batches, 1)


def save_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    try:
        from AWCANet import AWCANet
    except Exception as exc:
        raise RuntimeError(
            "Failed to import AWCANet. Please ensure dependencies are installed: "
            "torch, torchvision, timm, einops, dropblock. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        print("[train_awca] CUDA not available, fallback to CPU (slow).")
    device = torch.device("cuda" if use_cuda else "cpu")
    use_amp = bool(args.amp and use_cuda)

    mean, std = load_mean_std(args.data_root)
    train_loader, val_loader = build_dataloaders(
        data_root=args.data_root,
        tile_size=args.tile_size,
        mean=mean,
        std=std,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    pvt_path = None
    if not args.disable_pretrained:
        pvt_path = ensure_pvt_weights(args.pvt_weights)

    model = AWCANet(pvt_path=pvt_path).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = GradScaler(enabled=use_amp)

    work_dir = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = work_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = work_dir / "train_log.jsonl"

    start_epoch = 1
    best_f1 = -1.0

    if args.resume is not None:
        ckpt = torch.load(str(args.resume), map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_f1 = float(ckpt.get("best_f1", -1.0))
        print(f"[train_awca] Resume from {args.resume}, start_epoch={start_epoch}, best_f1={best_f1:.6f}")

    config_dump = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "device": str(device),
        "use_amp": use_amp,
        "mean": mean,
        "std": std,
        "train_size": len(train_loader.dataset),
        "val_size": len(val_loader.dataset),
    }
    save_json(work_dir / "config.json", config_dump)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
        )
        val_loss, val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            threshold=args.threshold,
        )
        scheduler.step()

        cur_lr = float(optimizer.param_groups[0]["lr"])
        elapsed = time.time() - t0
        line = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_precision": val_metrics.precision,
            "val_recall": val_metrics.recall,
            "val_f1": val_metrics.f1,
            "val_iou": val_metrics.iou,
            "val_oa": val_metrics.oa,
            "lr": cur_lr,
            "sec": elapsed,
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_f1": best_f1,
            "mean": mean,
            "std": std,
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        }
        torch.save(ckpt, ckpt_dir / "last.pth")

        improved = val_metrics.f1 > best_f1
        if improved:
            best_f1 = val_metrics.f1
            ckpt["best_f1"] = best_f1
            torch.save(ckpt, ckpt_dir / "best.pth")

        print(
            f"[Epoch {epoch:03d}/{args.epochs}] "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"val_f1={val_metrics.f1:.6f} val_iou={val_metrics.iou:.6f} "
            f"lr={cur_lr:.2e} {'*best*' if improved else ''}"
        )

    print(f"[train_awca] Done. Best Val F1={best_f1:.6f}, checkpoints at {ckpt_dir}")


if __name__ == "__main__":
    main()
