import os
import json
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader

from data.dataset import UrbanSARFloodsDataset, collate_fn
from data.sampler import build_train_sampler
from losses.hierarchical_loss import RegionAwareHierLoss
from models.region_aware_hier_model import RegionAwareHierarchicalPanFlood


def build_datasets(cfg: Dict, project_root: str) -> Tuple[UrbanSARFloodsDataset, UrbanSARFloodsDataset]:
    data_cfg = cfg["data"]

    train_ds = UrbanSARFloodsDataset(
        data_root=os.path.normpath(os.path.join(project_root, data_cfg["root"])),
        split_file=data_cfg["train_split"],
        input_mode=data_cfg["input_mode"],
        mean=data_cfg["mean"],
        std=data_cfg["std"],
        crop_size=data_cfg.get("crop_size", 252),
        random_crop=True,
        random_hflip=True,
        random_vflip=True,
        ignore_index=data_cfg.get("ignore_index", 255),
        auto_label_mapping=data_cfg.get("auto_label_mapping", False),
        seed=cfg.get("seed", 42),
    )

    val_ds = UrbanSARFloodsDataset(
        data_root=os.path.normpath(os.path.join(project_root, data_cfg["root"])),
        split_file=data_cfg.get("val_split", data_cfg["train_split"]),
        input_mode=data_cfg["input_mode"],
        mean=data_cfg["mean"],
        std=data_cfg["std"],
        crop_size=data_cfg.get("crop_size", 252),
        random_crop=False,
        random_hflip=False,
        random_vflip=False,
        ignore_index=data_cfg.get("ignore_index", 255),
        auto_label_mapping=data_cfg.get("auto_label_mapping", False),
        seed=cfg.get("seed", 42),
    )
    return train_ds, val_ds


def _load_hard_scores(train_ds: UrbanSARFloodsDataset, hard_score_file: str, project_root: str):
    if not hard_score_file:
        return None
    hpath = hard_score_file if os.path.isabs(hard_score_file) else os.path.join(project_root, hard_score_file)
    with open(hpath, "r", encoding="utf-8") as f:
        obj = json.load(f)

    # 支持两种格式：
    # 1) {"by_gt_name": {"xxx_GT.tif": score}}
    # 2) {"scores": [..]} (与 train_ds.pairs 顺序一致)
    if "scores" in obj and isinstance(obj["scores"], list):
        scores = obj["scores"]
        if len(scores) != len(train_ds.pairs):
            raise ValueError(f"hard_score_file length mismatch: {len(scores)} vs {len(train_ds.pairs)}")
        return [float(x) for x in scores]

    score_map = obj.get("by_gt_name", {})
    hard_scores = []
    for _, gt_path in train_ds.pairs:
        hard_scores.append(float(score_map.get(os.path.basename(gt_path), 0.0)))
    return hard_scores


def build_loaders(cfg: Dict, train_ds: UrbanSARFloodsDataset, val_ds: UrbanSARFloodsDataset, project_root: str):
    train_cfg = cfg["train"]
    sampler_cfg = cfg.get("sampler", {})

    hard_score_file = sampler_cfg.get("hard_score_file", "")
    hard_scores = _load_hard_scores(train_ds, hard_score_file, project_root=project_root)

    sampler = build_train_sampler(
        train_ds,
        enable_urban_oversample=sampler_cfg.get("enable_urban_oversample", True),
        urban_weight=sampler_cfg.get("urban_weight", 4.0),
        open_weight=sampler_cfg.get("open_weight", 1.5),
        hard_scores=hard_scores,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.get("val_batch_size", 1),
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader


def build_model(cfg: Dict) -> RegionAwareHierarchicalPanFlood:
    return RegionAwareHierarchicalPanFlood(cfg)


def build_loss(cfg: Dict) -> RegionAwareHierLoss:
    return RegionAwareHierLoss(cfg)


def build_optimizer(cfg: Dict, model: RegionAwareHierarchicalPanFlood):
    opt_cfg = cfg["optim"]
    wd = float(opt_cfg.get("weight_decay", 0.01))

    adapter_params = [p for _, p in model.backbone.named_adapter_parameters() if p.requires_grad]
    backbone_params = [p for _, p in model.backbone.named_backbone_parameters() if p.requires_grad]
    other_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("backbone.") or n.startswith("source_role_embed"):
            continue
        other_params.append(p)

    param_groups = [
        {"params": adapter_params, "lr": float(opt_cfg.get("lr_adapter", 1e-4)), "weight_decay": wd},
        {"params": other_params, "lr": float(opt_cfg.get("lr_heads", 1e-4)), "weight_decay": wd},
        {"params": backbone_params, "lr": float(opt_cfg.get("lr_backbone", 1e-5)), "weight_decay": wd},
    ]
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999))

    sch_cfg = cfg.get("scheduler", {})
    if sch_cfg.get("name", "cosine") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg["train"]["epochs"]))
    else:
        scheduler = None

    return optimizer, scheduler
