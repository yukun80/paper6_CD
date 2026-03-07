from __future__ import annotations

import argparse
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from tqdm import tqdm
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None
try:
    from torch.amp import GradScaler as AmpGradScaler
    from torch.amp import autocast as amp_autocast

    _USE_TORCH_AMP = True
except ImportError:  # torch==2.0.x 兼容分支
    from torch.cuda.amp import GradScaler as AmpGradScaler
    from torch.cuda.amp import autocast as amp_autocast

    _USE_TORCH_AMP = False

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

"""
python exp_template/train.py --config-file exp_template/config/resnet_fcn.yaml
python exp_template/train.py --config-file exp_template/config/unet.yaml
python exp_template/train.py --config-file exp_template/config/deeplabv3plus.yaml
python exp_template/train.py --config-file exp_template/config/pspnet.yaml
python exp_template/train.py --config-file exp_template/config/swin_uperlite.yaml
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("UrbanSARFloods CH12 segmentation training")
    parser.add_argument("--config-file", type=str, required=True, help="YAML 配置文件")
    parser.add_argument("--work-dir", type=str, default="", help="可选，覆盖配置中的 work_dir.root")
    parser.add_argument("--resume", type=str, default="", help="可选，继续训练的 checkpoint 路径")
    return parser.parse_args()


def build_grad_scaler(enabled: bool):
    """兼容 torch.amp 与 torch.cuda.amp 的 GradScaler 初始化差异。"""

    if _USE_TORCH_AMP:
        try:
            return AmpGradScaler(device="cuda", enabled=enabled)
        except TypeError:
            return AmpGradScaler(enabled=enabled)
    return AmpGradScaler(enabled=enabled)


def build_amp_context(device: torch.device, enabled: bool):
    """返回当前环境可用的 autocast 上下文。"""

    if not enabled:
        return nullcontext()
    if _USE_TORCH_AMP:
        return amp_autocast(device_type=device.type, enabled=True)
    if device.type == "cuda":
        return amp_autocast(enabled=True)
    return nullcontext()


def run_one_epoch(
    model: torch.nn.Module,
    loader,
    criterion,
    device: torch.device,
    epoch: int,
    max_epochs: int,
    phase_name: str,
    train_mode: bool,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[Any],
    amp_enabled: bool,
    current_lr: float,
    show_progress: bool,
    progress_leave: bool,
    ignore_index: int,
    num_classes: int,
    aux_loss_weight: float,
    grad_clip_norm: float,
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

    pbar = tqdm(
        loader,
        desc=f"[{phase_name}] Epoch {epoch}/{max_epochs}",
        leave=progress_leave,
        dynamic_ncols=True,
        disable=(not show_progress),
    )

    for images, labels, _ in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if train_mode:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)

        with build_amp_context(device=device, enabled=amp_enabled):
            output = model(images)
            logits, aux_logits = extract_logits_and_aux(output)
            main_loss = criterion(logits, labels)
            loss = main_loss
            if aux_logits is not None:
                loss = loss + aux_loss_weight * criterion(aux_logits, labels)

        if train_mode:
            assert scaler is not None
            scaler.scale(loss).backward()
            if grad_clip_norm > 0:
                # AMP 下裁剪前需要先反缩放梯度，避免裁剪阈值失真。
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

        pred = torch.argmax(logits, dim=1)
        meter.update(pred, labels)

        loss_sum += float(loss.item())
        batch_count += 1
        avg_loss = loss_sum / max(batch_count, 1)
        if train_mode:
            pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{current_lr:.2e}")
        else:
            pbar.set_postfix(loss=f"{avg_loss:.4f}")

    metrics = meter.compute(pos_classes=pos_classes)
    metrics["loss"] = loss_sum / max(batch_count, 1)
    return metrics["loss"], metrics


def is_better(candidate: float, current: float, mode: str = "max") -> bool:
    if mode == "max":
        return candidate > current
    if mode == "min":
        return candidate < current
    raise ValueError(f"selection_mode 仅支持 max/min，当前: {mode}")


def setup_tensorboard_layout(tb_writer: Optional[Any], layout_style: str) -> None:
    """注册 TensorBoard custom_scalars 排版，避免所有曲线单列堆叠。"""

    if tb_writer is None:
        return
    style = str(layout_style).strip().lower()
    if style == "off":
        return

    layout_map = {
        "grouped": {
            "Loss": {
                "train_vs_val_loss": ["Multiline", ["train/loss", "val/loss"]],
            },
            "Validation Metrics": {
                "iou_panel": ["Multiline", ["val/pos_mIoU", "val/mIoU"]],
                "oa_panel": ["Multiline", ["val/OA"]],
            },
        },
        "minimal": {
            "Loss": {
                "train_vs_val_loss": ["Multiline", ["train/loss", "val/loss"]],
            },
        },
        "full": {
            "Loss": {
                "train_vs_val_loss": ["Multiline", ["train/loss", "val/loss"]],
            },
            "Validation Metrics": {
                "iou_panel": ["Multiline", ["val/pos_mIoU", "val/mIoU"]],
                "oa_panel": ["Multiline", ["val/OA"]],
                "prf_panel": ["Multiline", ["val/mPrecision", "val/mRecall", "val/mF1"]],
            },
        },
    }
    chosen_layout = layout_map.get(style, layout_map["grouped"])
    tb_writer.add_custom_scalars(chosen_layout)


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
    tb_writer = None

    save_yaml(run_dirs["logs"] / "resolved_config.yaml", cfg)
    log_line(log_file, f"配置文件: {args.config_file}")
    log_line(log_file, f"数据根目录: {cfg['data']['root']}")
    log_line(log_file, f"输出目录: {run_dirs['root']}")

    tb_cfg = cfg.get("train", {}).get("tensorboard", {})
    tb_enabled = bool(tb_cfg.get("enabled", True))
    tb_log_subdir = str(tb_cfg.get("log_subdir", "tb"))
    tb_layout_style = str(tb_cfg.get("layout_style", "grouped"))
    if tb_enabled:
        if SummaryWriter is None:
            log_line(log_file, "TensorBoard 未安装，跳过曲线记录。可安装 tensorboard 后启用。")
        else:
            tb_dir = run_dirs["logs"] / tb_log_subdir
            tb_dir.mkdir(parents=True, exist_ok=True)
            tb_writer = SummaryWriter(log_dir=str(tb_dir))
            setup_tensorboard_layout(tb_writer=tb_writer, layout_style=tb_layout_style)
            log_line(log_file, f"TensorBoard 日志目录: {tb_dir}")

    train_loader, val_loader = build_dataloaders(cfg["data"], cfg["train"])
    log_line(log_file, f"train samples={len(train_loader.dataset)}, val samples={len(val_loader.dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.get("runtime", {}).get("require_cuda", False) and device.type != "cuda":
        raise RuntimeError("配置要求 CUDA，但当前不可用")
    log_line(log_file, f"device={device.type}, amp={cfg.get('train', {}).get('amp', True)}")

    model = build_model(cfg["model"]).to(device)

    criterion = build_criterion(cfg.get("loss", {}), ignore_index=int(cfg["data"].get("ignore_index", 255))).to(device)
    optimizer = build_optimizer(model, cfg.get("optim", {}))

    max_epochs = int(cfg.get("train", {}).get("max_epochs", 80))
    scheduler = build_scheduler(optimizer, cfg.get("scheduler", {}), max_epochs=max_epochs)

    amp_enabled = bool(cfg.get("train", {}).get("amp", True) and device.type == "cuda")
    scaler = build_grad_scaler(enabled=amp_enabled)

    selection_metric = str(cfg.get("eval", {}).get("selection_metric", "pos_mIoU"))
    selection_mode = str(cfg.get("eval", {}).get("selection_mode", "max")).lower()
    pos_classes = tuple(cfg.get("eval", {}).get("pos_classes", [1, 2]))
    show_progress = bool(cfg.get("train", {}).get("show_progress", True))
    progress_leave = bool(cfg.get("train", {}).get("progress_leave", False))
    aux_loss_weight = float(cfg.get("loss", {}).get("aux_loss_weight", 0.4))
    grad_clip_norm = float(cfg.get("train", {}).get("grad_clip_norm", 0.0))
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
            epoch=epoch,
            max_epochs=max_epochs,
            phase_name="train",
            train_mode=True,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
            current_lr=lr,
            show_progress=show_progress,
            progress_leave=progress_leave,
            ignore_index=ignore_index,
            num_classes=num_classes,
            aux_loss_weight=aux_loss_weight,
            grad_clip_norm=grad_clip_norm,
            pos_classes=pos_classes,
        )

        with torch.no_grad():
            val_loss, val_metrics = run_one_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                epoch=epoch,
                max_epochs=max_epochs,
                phase_name="val",
                train_mode=False,
                optimizer=None,
                scaler=None,
                amp_enabled=amp_enabled,
                current_lr=lr,
                show_progress=show_progress,
                progress_leave=progress_leave,
                ignore_index=ignore_index,
                num_classes=num_classes,
                aux_loss_weight=aux_loss_weight,
                grad_clip_norm=0.0,
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

        if tb_writer is not None:
            tb_writer.add_scalar("train/loss", train_loss, epoch)
            tb_writer.add_scalar("val/loss", val_loss, epoch)
            if "pos_mIoU" in val_metrics:
                tb_writer.add_scalar("val/pos_mIoU", float(val_metrics["pos_mIoU"]), epoch)
            if "mIoU" in val_metrics:
                tb_writer.add_scalar("val/mIoU", float(val_metrics["mIoU"]), epoch)
            # OA 字段优先使用新增键，若不存在则回退 overall_acc。
            if "OA" in val_metrics:
                tb_writer.add_scalar("val/OA", float(val_metrics["OA"]), epoch)
            elif "overall_acc" in val_metrics:
                tb_writer.add_scalar("val/OA", float(val_metrics["overall_acc"]), epoch)
            tb_writer.flush()

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
    if tb_writer is not None:
        tb_writer.close()
    log_line(log_file, f"训练完成，总耗时 {cost_min:.2f} 分钟")


if __name__ == "__main__":
    main()
