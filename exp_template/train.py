from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.amp import GradScaler, autocast

from dataloader import build_dataloaders
from utils import (
    SegMetricMeter,
    build_criterion,
    build_model,
    build_optimizer,
    build_scheduler,
    extract_logits_and_aux,
    load_config,
    log_line,
    make_run_dirs,
    resolve_path,
    save_json,
    save_yaml,
    set_torch_home,
    setup_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("UrbanSARFloods CH12 segmentation training")
    parser.add_argument("--config-file", type=str, required=True, help="YAML 配置文件")
    parser.add_argument("--work-dir", type=str, default="", help="可选，覆盖配置中的 work_dir.root")
    parser.add_argument("--resume", type=str, default="", help="可选，继续训练的 checkpoint 路径")
    return parser.parse_args()


def run_one_epoch(
    model: torch.nn.Module,
    loader,
    criterion,
    device: torch.device,
    train_mode: bool,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[GradScaler],
    amp_enabled: bool,
    ignore_index: int,
    num_classes: int,
    aux_loss_weight: float,
    pos_classes,
) -> Tuple[float, Dict[str, float]]:
    """统一训练/验证单轮逻辑，确保指标口径一致。"""

    if train_mode:
        model.train()
    else:
        model.eval()

    meter = SegMetricMeter(num_classes=num_classes, ignore_index=ignore_index)
    loss_sum = 0.0
    batch_count = 0

    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if train_mode:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=amp_enabled):
            output = model(images)
            logits, aux_logits = extract_logits_and_aux(output)
            main_loss = criterion(logits, labels)
            loss = main_loss
            if aux_logits is not None:
                loss = loss + aux_loss_weight * criterion(aux_logits, labels)

        if train_mode:
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        pred = torch.argmax(logits, dim=1)
        meter.update(pred, labels)

        loss_sum += float(loss.item())
        batch_count += 1

    metrics = meter.compute(pos_classes=pos_classes)
    metrics["loss"] = loss_sum / max(batch_count, 1)
    return metrics["loss"], metrics


def is_better(candidate: float, current: float, mode: str = "max") -> bool:
    if mode == "max":
        return candidate > current
    if mode == "min":
        return candidate < current
    raise ValueError(f"selection_mode 仅支持 max/min，当前: {mode}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config_file)

    project_root = str(Path(__file__).resolve().parents[1])
    set_torch_home(project_root)

    seed = int(cfg.get("train", {}).get("seed", 42))
    setup_seed(seed=seed, deterministic=bool(cfg.get("train", {}).get("deterministic", False)))

    # 将配置中的相对路径解析为工程绝对路径
    cfg["data"]["root"] = resolve_path(cfg["data"]["root"], project_root)

    run_dirs = make_run_dirs(cfg, project_root=project_root, work_dir_override=args.work_dir)
    log_file = run_dirs["logs"] / "train.log"
    history_file = run_dirs["logs"] / "history.json"

    save_yaml(run_dirs["logs"] / "resolved_config.yaml", cfg)
    log_line(log_file, f"配置文件: {args.config_file}")
    log_line(log_file, f"数据根目录: {cfg['data']['root']}")
    log_line(log_file, f"输出目录: {run_dirs['root']}")

    train_loader, val_loader = build_dataloaders(cfg["data"], cfg["train"])
    log_line(log_file, f"train samples={len(train_loader.dataset)}, val samples={len(val_loader.dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.get("runtime", {}).get("require_cuda", False) and device.type != "cuda":
        raise RuntimeError("配置要求 CUDA，但当前不可用")

    model = build_model(cfg["model"]).to(device)

    criterion = build_criterion(cfg.get("loss", {}), ignore_index=int(cfg["data"].get("ignore_index", 255))).to(device)
    optimizer = build_optimizer(model, cfg.get("optim", {}))

    max_epochs = int(cfg.get("train", {}).get("max_epochs", 80))
    scheduler = build_scheduler(optimizer, cfg.get("scheduler", {}), max_epochs=max_epochs)

    amp_enabled = bool(cfg.get("train", {}).get("amp", True) and device.type == "cuda")
    scaler = GradScaler(device="cuda", enabled=amp_enabled)

    selection_metric = str(cfg.get("eval", {}).get("selection_metric", "pos_mIoU"))
    selection_mode = str(cfg.get("eval", {}).get("selection_mode", "max")).lower()
    pos_classes = tuple(cfg.get("eval", {}).get("pos_classes", [1, 2]))
    aux_loss_weight = float(cfg.get("loss", {}).get("aux_loss_weight", 0.4))
    num_classes = int(cfg["model"].get("num_classes", 3))
    ignore_index = int(cfg["data"].get("ignore_index", 255))

    start_epoch = 1
    best_metric = -float("inf") if selection_mode == "max" else float("inf")
    history = []

    resume_path = args.resume or cfg.get("train", {}).get("resume_from", "")
    if resume_path:
        resume_path = resolve_path(resume_path, project_root)
        ckpt = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_metric = float(ckpt.get("best_metric", best_metric))
        log_line(log_file, f"已从断点恢复: {resume_path} (start_epoch={start_epoch})")

    save_every = int(cfg.get("train", {}).get("save_every", 10))

    t0 = time.time()
    for epoch in range(start_epoch, max_epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        log_line(log_file, f"Epoch [{epoch}/{max_epochs}] lr={lr:.6e}")

        train_loss, train_metrics = run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            train_mode=True,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
            ignore_index=ignore_index,
            num_classes=num_classes,
            aux_loss_weight=aux_loss_weight,
            pos_classes=pos_classes,
        )

        with torch.no_grad():
            val_loss, val_metrics = run_one_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                train_mode=False,
                optimizer=None,
                scaler=None,
                amp_enabled=amp_enabled,
                ignore_index=ignore_index,
                num_classes=num_classes,
                aux_loss_weight=aux_loss_weight,
                pos_classes=pos_classes,
            )

        if scheduler is not None:
            scheduler.step()

        record = {
            "epoch": epoch,
            "lr": lr,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)
        save_json(history_file, {"history": history})

        log_line(
            log_file,
            (
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_{selection_metric}={val_metrics.get(selection_metric, float('nan')):.6f}"
            ),
        )

        ckpt = {
            "epoch": epoch,
            "best_metric": best_metric,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "config": cfg,
        }

        latest_path = run_dirs["checkpoints"] / "last.pth"
        torch.save(ckpt, latest_path)

        if epoch % save_every == 0:
            torch.save(ckpt, run_dirs["checkpoints"] / f"epoch_{epoch:03d}.pth")

        cur_metric = float(val_metrics.get(selection_metric, float("nan")))
        if not math.isnan(cur_metric) and is_better(cur_metric, best_metric, selection_mode):
            best_metric = cur_metric
            ckpt["best_metric"] = best_metric
            torch.save(ckpt, run_dirs["checkpoints"] / "best.pth")
            log_line(log_file, f"更新 best checkpoint: {selection_metric}={best_metric:.6f}")

    cost_min = (time.time() - t0) / 60.0
    log_line(log_file, f"训练完成，总耗时 {cost_min:.2f} 分钟")


if __name__ == "__main__":
    main()
