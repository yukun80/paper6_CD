import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from dinov2.layers.attention import XFORMERS_AVAILABLE, XFORMERS_ENABLED
from urban_floods.dataset import UrbanSARFloodsSegDataset, collate_urban_floods
from urban_floods.metrics import multiclass_dice_loss, summarize_from_confusion, update_confusion_matrix
from urban_floods.model import PanopticonUrbanSeg


def parse_args():
    """解析训练脚本命令行参数。"""
    parser = argparse.ArgumentParser("Panopticon UrbanSARFloods segmentation training")
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="")
    return parser.parse_args()


def load_cfg(path: str) -> Dict:
    """加载 YAML 配置。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    """统一随机种子，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_xdict_to_device(x_dict: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """把样本字典中的张量全部搬到目标设备。"""
    return {k: v.to(device, non_blocking=True) for k, v in x_dict.items()}


def _resolve_path(path: str, base_dir: str) -> str:
    """将配置中的相对路径解析为绝对路径。"""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def _resolve_runtime(cfg: Dict) -> Tuple[bool, torch.dtype, bool]:
    """解析 AMP 配置，并返回 (amp开关, amp dtype, 是否使用GradScaler)。"""
    runtime_cfg = cfg.get("runtime", {})
    amp_cfg = runtime_cfg.get("amp", {})
    amp_enabled = bool(amp_cfg.get("enabled", True))
    amp_dtype_cfg = str(amp_cfg.get("dtype", "bf16")).lower()

    # 优先 bf16；若设备/环境不支持则回退 fp16。
    if amp_dtype_cfg == "bf16" and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16

    use_scaler = amp_enabled and amp_dtype == torch.float16
    return amp_enabled, amp_dtype, use_scaler


def _is_better(
    candidate: Dict[str, float],
    current_best: Optional[Dict[str, float]],
    metric: str,
    mode: str = "max",
    tie_breakers: Sequence[str] = ("IoU_2", "mIoU"),
) -> bool:
    """按主指标+次级指标比较两次验证结果。"""
    if current_best is None:
        return True

    if mode not in {"max", "min"}:
        raise ValueError(f"Unsupported selection mode: {mode}")
    default = -math.inf if mode == "max" else math.inf

    def _cmp_key(stats: Dict[str, float], key: str) -> float:
        v = stats.get(key, default)
        return float(v) if v is not None else default

    keys = [metric, *tie_breakers]
    for k in keys:
        c = _cmp_key(candidate, k)
        b = _cmp_key(current_best, k)
        if c == b:
            continue
        return c > b if mode == "max" else c < b
    return False


def _assert_cuda_runtime(cfg: Dict):
    """强校验运行环境：本脚本按 CUDA-only 设计。"""
    require_cuda = bool(cfg.get("runtime", {}).get("require_cuda", True))
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "This pipeline is configured as CUDA-only (runtime.require_cuda=true), "
            "but CUDA is not available."
        )
    if not XFORMERS_ENABLED or not XFORMERS_AVAILABLE:
        raise RuntimeError(
            "xformers is required in CUDA-only mode, but it is disabled/unavailable. "
            "Please install xformers and ensure XFORMERS_DISABLED is unset."
        )


def _cuda_sanity_check(
    model: PanopticonUrbanSeg,
    device: torch.device,
    data_cfg: Dict,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
):
    """用一个最小 dummy 前向检查 CUDA + AMP 路径是否可用。"""
    crop = int(data_cfg.get("crop_size", 252))
    chn_ids = torch.tensor(data_cfg["channel_ids"], device=device, dtype=torch.long).unsqueeze(0)
    time_ids = torch.tensor(data_cfg["time_ids"], device=device, dtype=torch.long).unsqueeze(0)
    imgs = torch.zeros((1, len(data_cfg["channel_ids"]), crop, crop), device=device, dtype=torch.float32)
    x_dict = {"imgs": imgs, "chn_ids": chn_ids, "time_ids": time_ids}

    model.eval()
    with torch.no_grad():
        amp_ctx = torch.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled) if amp_enabled else nullcontext()
        with amp_ctx:
            out = model(x_dict)
    if not torch.isfinite(out).all():
        raise RuntimeError("CUDA sanity check failed: model output contains NaN/Inf.")


def build_datasets(cfg: Dict, project_root: str):
    """构建训练/验证数据集。"""
    data_cfg = cfg["data"]
    common_kwargs = dict(
        data_root=_resolve_path(data_cfg["root"], project_root),
        channel_ids=data_cfg["channel_ids"],
        time_ids=data_cfg["time_ids"],
        mean=data_cfg["mean"],
        std=data_cfg["std"],
        ignore_index=data_cfg.get("ignore_index", 255),
        crop_size=data_cfg.get("crop_size", 252),
        seed=cfg.get("seed", 42),
    )
    train_ds = UrbanSARFloodsSegDataset(
        split_file=data_cfg["train_split"],
        random_hflip=True,
        random_vflip=True,
        random_crop=True,
        **common_kwargs,
    )
    val_ds = UrbanSARFloodsSegDataset(
        split_file=data_cfg["val_split"],
        random_hflip=False,
        random_vflip=False,
        random_crop=False,
        **common_kwargs,
    )
    return train_ds, val_ds


def build_optimizer(model: PanopticonUrbanSeg, cfg: Dict, train_backbone: bool):
    """按阶段构建优化器：冻结阶段只训 head，解冻阶段训 backbone+head。"""
    opt_cfg = cfg["optim"]
    wd = opt_cfg.get("weight_decay", 0.01)

    if train_backbone:
        params = [
            {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": opt_cfg["lr_backbone"]},
            {"params": [p for p in model.decode_head.parameters() if p.requires_grad], "lr": opt_cfg["lr_head"]},
        ]
    else:
        params = [{"params": [p for p in model.decode_head.parameters() if p.requires_grad], "lr": opt_cfg["lr_head"]}]
    return torch.optim.AdamW(params, betas=(0.9, 0.999), weight_decay=wd)


@torch.no_grad()
def evaluate(
    model: PanopticonUrbanSeg,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
    loss_cfg: Dict,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    positive_classes: Sequence[int] = (1, 2),
) -> Dict[str, float]:
    """验证阶段：计算 loss 与 mIoU/mF1。"""
    model.eval()
    conf = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
    ce_total, dice_total, n = 0.0, 0.0, 0

    ce_loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
    for x_dict, target, _ in loader:
        x_dict = move_xdict_to_device(x_dict, device)
        target = target.to(device, non_blocking=True)

        # 验证阶段与训练阶段保持相同 AMP 路径，避免评估/训练数值行为不一致。
        amp_ctx = torch.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled) if amp_enabled else nullcontext()
        with amp_ctx:
            logits = model(x_dict)
            ce = ce_loss(logits, target)
            dice = multiclass_dice_loss(
                logits, target, num_classes=num_classes, ignore_index=ignore_index, eps=loss_cfg.get("eps", 1e-6)
            )

        pred = logits.argmax(dim=1)
        conf = update_confusion_matrix(conf, pred, target, num_classes=num_classes, ignore_index=ignore_index)

        ce_total += ce.item()
        dice_total += dice.item()
        n += 1

    m = summarize_from_confusion(conf, positive_classes=positive_classes)
    m["ce_loss"] = ce_total / max(n, 1)
    m["dice_loss"] = dice_total / max(n, 1)
    m["total_loss"] = loss_cfg["ce_weight"] * m["ce_loss"] + loss_cfg["dice_weight"] * m["dice_loss"]
    return m


def main():
    """训练入口。"""
    args = parse_args()
    cfg = load_cfg(args.config_file)
    _assert_cuda_runtime(cfg)
    set_seed(cfg.get("seed", 42))
    project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

    runtime_cfg = cfg.get("runtime", {})
    if bool(runtime_cfg.get("allow_tf32", True)):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    amp_enabled, amp_dtype, use_scaler = _resolve_runtime(cfg)
    device = torch.device("cuda")
    print(
        f"[runtime] device={device}, xformers_enabled={XFORMERS_ENABLED and XFORMERS_AVAILABLE}, "
        f"amp_enabled={amp_enabled}, amp_dtype={amp_dtype}, scaler={use_scaler}"
    )

    # 保存目录：优先命令行覆盖，其次配置文件。
    output_dir = args.output_dir or cfg["train"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "resolved_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    train_ds, val_ds = build_datasets(cfg, project_root=project_root)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        collate_fn=collate_urban_floods,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"].get("val_batch_size", 1),
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        collate_fn=collate_urban_floods,
        drop_last=False,
    )

    model = PanopticonUrbanSeg(
        ckpt_path=_resolve_path(cfg["model"]["checkpoint_path"], project_root),
        num_classes=cfg["model"]["num_classes"],
        block_indices=cfg["model"]["block_indices"],
        decode_channels=cfg["model"]["decode_channels"],
        use_time_embed=cfg["model"].get("use_time_embed", True),
    ).to(device)
    print("Checkpoint load msg:", model.load_msg)
    # 在真正迭代前先做一次模型可用性检查，尽早暴露环境问题。
    _cuda_sanity_check(
        model=model,
        device=device,
        data_cfg=cfg["data"],
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
    )

    n_classes = cfg["model"]["num_classes"]
    ignore_index = cfg["data"].get("ignore_index", 255)
    positive_classes = cfg["train"].get("positive_classes", [1, 2])
    selection_metric = str(cfg["train"].get("selection_metric", "pos_mIoU"))
    selection_mode = str(cfg["train"].get("selection_mode", "max")).lower()
    tie_breakers = cfg["train"].get("selection_tie_breakers", ["IoU_2", "mIoU"])

    # 类别权重来自训练集标签直方图，缓解类别不平衡。
    if cfg["loss"].get("auto_class_weights", True):
        hist = train_ds.compute_class_histogram(n_classes=n_classes)
        hist = hist + 1.0
        weights = hist.sum() / hist
        weights = (weights / weights.mean()).float().to(device)
    else:
        weights = None
    ce_loss = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=ignore_index)
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    freeze_epochs = int(cfg["train"].get("freeze_backbone_epochs", 0))
    epochs = int(cfg["train"]["epochs"])

    # 训练阶段1：冻结 backbone（若 freeze_epochs > 0）。
    model.set_backbone_trainable(False if freeze_epochs > 0 else True)
    train_backbone = freeze_epochs <= 0
    optimizer = build_optimizer(model, cfg, train_backbone=train_backbone)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    best_stats = None
    history = []
    global_step = 0

    for epoch in range(epochs):
        tic = time.time()
        # 到达指定轮次后进入阶段2：解冻 backbone。
        if epoch == freeze_epochs and freeze_epochs > 0:
            model.set_backbone_trainable(True)
            train_backbone = True
            optimizer = build_optimizer(model, cfg, train_backbone=True)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - epoch))
            print(f"[Epoch {epoch}] unfreeze backbone and rebuild optimizer.")

        model.train()
        train_loss = 0.0
        train_invalid = 0.0
        count = 0

        for x_dict, target, invalid_ratio in train_loader:
            x_dict = move_xdict_to_device(x_dict, device)
            target = target.to(device, non_blocking=True)
            invalid_ratio = invalid_ratio.to(device, non_blocking=True)

            # 前向与 loss 在 autocast 中执行，兼顾速度和显存。
            amp_ctx = torch.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled) if amp_enabled else nullcontext()
            with amp_ctx:
                logits = model(x_dict)
                loss_ce = ce_loss(logits, target)
                loss_dice = multiclass_dice_loss(
                    logits, target, num_classes=n_classes, ignore_index=ignore_index, eps=cfg["loss"].get("eps", 1e-6)
                )
                loss = cfg["loss"]["ce_weight"] * loss_ce + cfg["loss"]["dice_weight"] * loss_dice

            optimizer.zero_grad(set_to_none=True)
            # fp16 走 GradScaler；bf16/FP32 直接反向。
            if use_scaler:
                scaler.scale(loss).backward()
                if cfg["train"].get("grad_clip_norm", 0) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip_norm"])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if cfg["train"].get("grad_clip_norm", 0) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip_norm"])
                optimizer.step()

            train_loss += loss.item()
            train_invalid += invalid_ratio.mean().item()
            count += 1
            global_step += 1

            if global_step % cfg["train"].get("log_interval", 50) == 0:
                print(
                    f"[Step {global_step}] loss={loss.item():.4f} ce={loss_ce.item():.4f} "
                    f"dice={loss_dice.item():.4f} invalid_ratio={invalid_ratio.mean().item():.4f}"
                )

        scheduler.step()

        # 每个 epoch 后进行验证。
        val_stats = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            num_classes=n_classes,
            ignore_index=ignore_index,
            loss_cfg=cfg["loss"],
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            positive_classes=positive_classes,
        )
        epoch_stats = {
            "epoch": epoch,
            "train_loss": train_loss / max(count, 1),
            "train_invalid_ratio": train_invalid / max(count, 1),
            **val_stats,
            "seconds": time.time() - tic,
        }
        history.append(epoch_stats)
        print(
            f"[Epoch {epoch}] train_loss={epoch_stats['train_loss']:.4f} "
            f"val_pos_mIoU={epoch_stats['pos_mIoU']:.4f} "
            f"val_IoU_1={epoch_stats['IoU_1']:.4f} val_IoU_2={epoch_stats['IoU_2']:.4f} "
            f"val_mIoU={epoch_stats['mIoU']:.4f} val_mF1={epoch_stats['mF1']:.4f} "
            f"val_loss={epoch_stats['total_loss']:.4f}"
        )

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if use_scaler else None,
            "stats": epoch_stats,
            "config": cfg,
        }
        torch.save(ckpt, os.path.join(output_dir, "last.pth"))

        # 依据配置中的 selection_metric 保存 best checkpoint（默认 pos_mIoU）。
        if _is_better(
            candidate=val_stats,
            current_best=best_stats,
            metric=selection_metric,
            mode=selection_mode,
            tie_breakers=tie_breakers,
        ):
            best_stats = dict(val_stats)
            torch.save(ckpt, os.path.join(output_dir, "best.pth"))
            print(f"[Epoch {epoch}] new best {selection_metric}={val_stats.get(selection_metric, float('nan')):.4f}")

        with open(os.path.join(output_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    best_val = best_stats.get(selection_metric, float("nan")) if best_stats is not None else float("nan")
    print(f"Training done. Best {selection_metric}={best_val:.4f}")


if __name__ == "__main__":
    main()
