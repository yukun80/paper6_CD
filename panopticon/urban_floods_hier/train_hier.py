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
from urban_floods_hier.dataset import UrbanSARFloodsHierDataset, collate_urban_floods_hier
from urban_floods_hier.losses import HierarchicalPanFloodLoss
from urban_floods_hier.metrics import summarize_from_confusion, update_confusion_matrix
from urban_floods_hier.model import HierarchicalPanFloodAdapter

"""
PYTHONPATH=panopticon python panopticon/urban_floods_hier/train_hier.py \
    --config-file panopticon/configs/urban_floods_hier_seg.yaml
"""

def parse_args():
    """解析训练脚本命令行参数。"""
    parser = argparse.ArgumentParser("Hierarchical PanFlood-Adapter training")
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
    return {k: v.to(device, non_blocking=True) for k, v in x_dict.items()}


def move_ydict_to_device(y_dict: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in y_dict.items()}


def _resolve_path(path: str, base_dir: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def _resolve_runtime(cfg: Dict) -> Tuple[bool, torch.dtype, bool]:
    runtime_cfg = cfg.get("runtime", {})
    amp_cfg = runtime_cfg.get("amp", {})
    amp_enabled = bool(amp_cfg.get("enabled", True))
    amp_dtype_cfg = str(amp_cfg.get("dtype", "bf16")).lower()

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
    require_cuda = bool(cfg.get("runtime", {}).get("require_cuda", True))
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "This pipeline is configured as CUDA-only (runtime.require_cuda=true), but CUDA is not available."
        )
    if require_cuda and (not XFORMERS_ENABLED or not XFORMERS_AVAILABLE):
        raise RuntimeError(
            "xformers is required in CUDA-only mode, but it is disabled/unavailable. "
            "Please install xformers and ensure XFORMERS_DISABLED is unset."
        )


def _cuda_sanity_check(
    model: HierarchicalPanFloodAdapter,
    device: torch.device,
    data_cfg: Dict,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
):
    """在真正训练前做一次最小前向检查，提前暴露配置问题。"""
    crop = int(data_cfg.get("crop_size", 252))
    chn = len(data_cfg["channel_ids"])
    x_dict = {
        "imgs": torch.zeros((1, chn, crop, crop), device=device, dtype=torch.float32),
        "chn_ids": torch.tensor(data_cfg["channel_ids"], device=device, dtype=torch.long).unsqueeze(0),
        "time_ids": torch.tensor(data_cfg["time_ids"], device=device, dtype=torch.long).unsqueeze(0),
        "feature_type_ids": torch.tensor(data_cfg["feature_type_ids"], device=device, dtype=torch.long).unsqueeze(0),
        "temporal_role_ids": torch.tensor(data_cfg["temporal_role_ids"], device=device, dtype=torch.long).unsqueeze(0),
        "polarization_ids": torch.tensor(data_cfg["polarization_ids"], device=device, dtype=torch.long).unsqueeze(0),
    }

    model.eval()
    with torch.no_grad():
        amp_ctx = torch.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled) if amp_enabled else nullcontext()
        with amp_ctx:
            out = model(x_dict)
    if not torch.isfinite(out["final_probs"]).all():
        raise RuntimeError("CUDA sanity check failed: model output contains NaN/Inf.")


def build_datasets(cfg: Dict, project_root: str):
    """构建训练/验证数据集。"""
    data_cfg = cfg["data"]
    common_kwargs = dict(
        data_root=_resolve_path(data_cfg["root"], project_root),
        channel_ids=data_cfg["channel_ids"],
        time_ids=data_cfg["time_ids"],
        feature_type_ids=data_cfg["feature_type_ids"],
        temporal_role_ids=data_cfg["temporal_role_ids"],
        polarization_ids=data_cfg["polarization_ids"],
        mean=data_cfg["mean"],
        std=data_cfg["std"],
        main_label_dir=data_cfg.get("main_label_dir", "GT"),
        floodness_label_dir=data_cfg.get("floodness_label_dir", "GT_floodness"),
        flood_type_label_dir=data_cfg.get("flood_type_label_dir", "GT_flood_type"),
        sar_dir=data_cfg.get("sar_dir", "SAR"),
        ignore_index_main=data_cfg.get("ignore_index_main", 255),
        ignore_index_floodness=data_cfg.get("ignore_index_floodness", 255),
        ignore_index_flood_type=data_cfg.get("ignore_index_flood_type", 255),
        crop_size=data_cfg.get("crop_size", 252),
        seed=cfg.get("seed", 42),
    )

    train_ds = UrbanSARFloodsHierDataset(
        split_file=data_cfg["train_split"],
        random_hflip=True,
        random_vflip=True,
        random_crop=True,
        **common_kwargs,
    )
    val_ds = UrbanSARFloodsHierDataset(
        split_file=data_cfg["val_split"],
        random_hflip=False,
        random_vflip=False,
        random_crop=False,
        **common_kwargs,
    )
    return train_ds, val_ds


def _build_criterion(cfg: Dict, class_weights: Optional[torch.Tensor]) -> HierarchicalPanFloodLoss:
    loss_cfg = cfg["loss"]
    return HierarchicalPanFloodLoss(
        ignore_index_main=cfg["data"].get("ignore_index_main", 255),
        ignore_index_floodness=cfg["data"].get("ignore_index_floodness", 255),
        ignore_index_flood_type=cfg["data"].get("ignore_index_flood_type", 255),
        floodness_bce_weight=loss_cfg.get("floodness_bce_weight", 1.0),
        floodness_dice_weight=loss_cfg.get("floodness_dice_weight", 1.0),
        type_loss_kind=loss_cfg.get("type_loss_kind", "bce"),
        type_focal_alpha=loss_cfg.get("type_focal_alpha", 0.25),
        type_focal_gamma=loss_cfg.get("type_focal_gamma", 2.0),
        main_ce_weight=loss_cfg.get("main_ce_weight", 0.5),
        type_loss_weight=loss_cfg.get("type_loss_weight", 0.8),
        eps=loss_cfg.get("eps", 1e-6),
        main_ce_class_weights=class_weights,
    )


def build_optimizer(model: HierarchicalPanFloodAdapter, cfg: Dict):
    """按模块学习率构建优化器。"""
    opt_cfg = cfg["optim"]
    wd = opt_cfg.get("weight_decay", 0.01)
    param_groups = model.get_trainable_param_groups(
        lr_backbone=opt_cfg.get("lr_backbone", 1e-5),
        lr_metadata=opt_cfg.get("lr_metadata", opt_cfg.get("lr_head", 1e-4)),
        lr_input_adapter=opt_cfg.get("lr_input_adapter", opt_cfg.get("lr_head", 1e-4)),
        lr_neck=opt_cfg.get("lr_neck", opt_cfg.get("lr_head", 1e-4)),
        lr_heads=opt_cfg.get("lr_heads", opt_cfg.get("lr_head", 1e-4)),
        weight_decay=wd,
    )

    for g in param_groups:
        n_params = sum(p.numel() for p in g["params"])
        print(f"[optimizer] group={g['name']}, lr={g['lr']:.2e}, params={n_params}")

    return torch.optim.AdamW(param_groups, betas=(0.9, 0.999), weight_decay=0.0)


@torch.no_grad()
def evaluate(
    model: HierarchicalPanFloodAdapter,
    criterion: HierarchicalPanFloodLoss,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    positive_classes: Sequence[int] = (1, 2),
) -> Dict[str, float]:
    """验证阶段：计算层次损失与主任务分割指标。"""
    model.eval()
    conf = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)

    total_loss, floodness_loss, type_loss, main_loss, n = 0.0, 0.0, 0.0, 0.0, 0
    for x_dict, y_dict, _ in loader:
        x_dict = move_xdict_to_device(x_dict, device)
        y_dict = move_ydict_to_device(y_dict, device)

        amp_ctx = torch.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled) if amp_enabled else nullcontext()
        with amp_ctx:
            out = model(x_dict)
            loss_dict = criterion(
                floodness_logits=out["floodness_logits"],
                flood_type_logits=out["flood_type_logits"],
                final_logits=out["final_logits"],
                final_probs=out["final_probs"],
                main_label=y_dict["main_label"],
                floodness_label=y_dict["floodness_label"],
                flood_type_label=y_dict["flood_type_label"],
                valid_mask=y_dict["valid_mask"],
            )

        pred = out["final_logits"].argmax(dim=1)
        conf = update_confusion_matrix(
            conf,
            pred,
            y_dict["main_label"],
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

        total_loss += float(loss_dict["loss_total"].item())
        floodness_loss += float(loss_dict["loss_floodness"].item())
        type_loss += float(loss_dict["loss_type"].item())
        main_loss += float(loss_dict["loss_main"].item())
        n += 1

    m = summarize_from_confusion(conf, positive_classes=positive_classes)
    m["loss_total"] = total_loss / max(n, 1)
    m["loss_floodness"] = floodness_loss / max(n, 1)
    m["loss_type"] = type_loss / max(n, 1)
    m["loss_main"] = main_loss / max(n, 1)
    return m


def main():
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
        collate_fn=collate_urban_floods_hier,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"].get("val_batch_size", 1),
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        collate_fn=collate_urban_floods_hier,
        drop_last=False,
    )

    model = HierarchicalPanFloodAdapter(
        ckpt_path=_resolve_path(cfg["model"].get("checkpoint_path", ""), project_root)
        if cfg["model"].get("checkpoint_path", "")
        else None,
        block_indices=cfg["model"].get("block_indices", [3, 5, 7, 11]),
        fpn_dim=cfg["model"].get("fpn_dim", 256),
        ppm_scales=cfg["model"].get("ppm_scales", [1, 2, 3, 6]),
        head_hidden_channels=cfg["model"].get("head_hidden_channels", 128),
        use_time_embed=cfg["model"].get("use_time_embed", True),
        use_metadata_embed=cfg["model"].get("use_metadata_embed", True),
    ).to(device)
    print("Checkpoint load msg:", model.load_msg)

    _cuda_sanity_check(model=model, device=device, data_cfg=cfg["data"], amp_enabled=amp_enabled, amp_dtype=amp_dtype)

    n_classes = int(cfg["model"].get("num_classes", 3))
    ignore_index = int(cfg["data"].get("ignore_index_main", 255))
    positive_classes = cfg["train"].get("positive_classes", [1, 2])
    selection_metric = str(cfg["train"].get("selection_metric", "pos_mIoU"))
    selection_mode = str(cfg["train"].get("selection_mode", "max")).lower()
    tie_breakers = cfg["train"].get("selection_tie_breakers", ["IoU_2", "mIoU"])

    if cfg["loss"].get("auto_class_weights", True):
        hist = train_ds.compute_main_class_histogram(n_classes=n_classes) + 1.0
        class_weights = (hist.sum() / hist)
        class_weights = (class_weights / class_weights.mean()).float().to(device)
    else:
        class_weights = None

    criterion = _build_criterion(cfg, class_weights=class_weights)
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    freeze_epochs = int(cfg["train"].get("freeze_backbone_epochs", 0))
    unfreeze_last_n_blocks = int(cfg["train"].get("unfreeze_last_n_blocks", 5))
    epochs = int(cfg["train"]["epochs"])

    if freeze_epochs > 0:
        model.freeze_backbone()
        print(f"[Stage A] freeze backbone blocks for first {freeze_epochs} epochs.")
    else:
        model.unfreeze_backbone_last_n_blocks(unfreeze_last_n_blocks)
        print(f"[Stage B only] train with last {unfreeze_last_n_blocks} backbone blocks unfrozen.")

    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    best_stats = None
    history = []
    global_step = 0

    for epoch in range(epochs):
        tic = time.time()

        if epoch == freeze_epochs and freeze_epochs > 0:
            model.unfreeze_backbone_last_n_blocks(unfreeze_last_n_blocks)
            optimizer = build_optimizer(model, cfg)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - epoch))
            print(f"[Epoch {epoch}] switch to Stage B, unfreeze last {unfreeze_last_n_blocks} blocks.")

        model.train()
        running = {
            "loss_total": 0.0,
            "loss_floodness": 0.0,
            "loss_type": 0.0,
            "loss_main": 0.0,
            "invalid_ratio": 0.0,
        }
        count = 0

        for x_dict, y_dict, invalid_ratio in train_loader:
            x_dict = move_xdict_to_device(x_dict, device)
            y_dict = move_ydict_to_device(y_dict, device)
            invalid_ratio = invalid_ratio.to(device, non_blocking=True)

            amp_ctx = torch.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled) if amp_enabled else nullcontext()
            with amp_ctx:
                out = model(x_dict)
                loss_dict = criterion(
                    floodness_logits=out["floodness_logits"],
                    flood_type_logits=out["flood_type_logits"],
                    final_logits=out["final_logits"],
                    final_probs=out["final_probs"],
                    main_label=y_dict["main_label"],
                    floodness_label=y_dict["floodness_label"],
                    flood_type_label=y_dict["flood_type_label"],
                    valid_mask=y_dict["valid_mask"],
                )
                loss = loss_dict["loss_total"]

            optimizer.zero_grad(set_to_none=True)
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

            running["loss_total"] += float(loss_dict["loss_total"].item())
            running["loss_floodness"] += float(loss_dict["loss_floodness"].item())
            running["loss_type"] += float(loss_dict["loss_type"].item())
            running["loss_main"] += float(loss_dict["loss_main"].item())
            running["invalid_ratio"] += float(invalid_ratio.mean().item())

            count += 1
            global_step += 1

            if global_step % cfg["train"].get("log_interval", 50) == 0:
                print(
                    f"[Step {global_step}] total={loss_dict['loss_total'].item():.4f} "
                    f"floodness={loss_dict['loss_floodness'].item():.4f} "
                    f"type={loss_dict['loss_type'].item():.4f} "
                    f"main={loss_dict['loss_main'].item():.4f} "
                    f"invalid_ratio={invalid_ratio.mean().item():.4f}"
                )

        scheduler.step()

        val_stats = evaluate(
            model=model,
            criterion=criterion,
            loader=val_loader,
            device=device,
            num_classes=n_classes,
            ignore_index=ignore_index,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            positive_classes=positive_classes,
        )

        epoch_stats = {
            "epoch": epoch,
            "train_loss": running["loss_total"] / max(count, 1),
            "train_loss_floodness": running["loss_floodness"] / max(count, 1),
            "train_loss_type": running["loss_type"] / max(count, 1),
            "train_loss_main": running["loss_main"] / max(count, 1),
            "train_invalid_ratio": running["invalid_ratio"] / max(count, 1),
            **val_stats,
            "seconds": time.time() - tic,
        }
        history.append(epoch_stats)

        print(
            f"[Epoch {epoch}] train_loss={epoch_stats['train_loss']:.4f} "
            f"val_pos_mIoU={epoch_stats['pos_mIoU']:.4f} "
            f"val_IoU_1={epoch_stats['IoU_1']:.4f} val_IoU_2={epoch_stats['IoU_2']:.4f} "
            f"val_mIoU={epoch_stats['mIoU']:.4f} "
            f"val_loss={epoch_stats['loss_total']:.4f}"
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
