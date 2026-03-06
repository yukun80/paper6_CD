import argparse
import json
import os
from typing import Dict

import torch
import yaml
from torch.utils.data import DataLoader

from dinov2.layers.attention import XFORMERS_AVAILABLE, XFORMERS_ENABLED
from urban_floods_hier.dataset import UrbanSARFloodsHierDataset, collate_urban_floods_hier
from urban_floods_hier.train_hier import _build_criterion, _resolve_runtime, evaluate
from urban_floods_hier.model import HierarchicalPanFloodAdapter


def parse_args():
    """解析评估脚本参数。"""
    parser = argparse.ArgumentParser("Hierarchical PanFlood-Adapter evaluation")
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["val", "train"])
    parser.add_argument("--output", type=str, default="")
    return parser.parse_args()


def load_cfg(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path: str, base_dir: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def assert_cuda_runtime(cfg: Dict):
    require_cuda = bool(cfg.get("runtime", {}).get("require_cuda", True))
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "This pipeline is configured as CUDA-only (runtime.require_cuda=true), "
            "but CUDA is not available."
        )
    if require_cuda and (not XFORMERS_ENABLED or not XFORMERS_AVAILABLE):
        raise RuntimeError(
            "xformers is required in CUDA-only mode, but it is disabled/unavailable. "
            "Please install xformers and ensure XFORMERS_DISABLED is unset."
        )


def main():
    args = parse_args()
    cfg = load_cfg(args.config_file)
    assert_cuda_runtime(cfg)
    project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

    runtime_cfg = cfg.get("runtime", {})
    if bool(runtime_cfg.get("allow_tf32", True)):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    amp_enabled, amp_dtype, _ = _resolve_runtime(cfg)

    data_cfg = cfg["data"]
    split_file = data_cfg["val_split"] if args.split == "val" else data_cfg["train_split"]
    ds = UrbanSARFloodsHierDataset(
        data_root=resolve_path(data_cfg["root"], project_root),
        split_file=split_file,
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
        random_hflip=False,
        random_vflip=False,
        random_crop=False,
    )
    loader = DataLoader(
        ds,
        batch_size=cfg["train"].get("val_batch_size", 1),
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        collate_fn=collate_urban_floods_hier,
        drop_last=False,
    )

    device = torch.device("cuda")
    print(
        f"[runtime] device={device}, xformers_enabled={XFORMERS_ENABLED and XFORMERS_AVAILABLE}, "
        f"amp_enabled={amp_enabled}, amp_dtype={amp_dtype}"
    )

    model = HierarchicalPanFloodAdapter(
        ckpt_path=resolve_path(cfg["model"].get("checkpoint_path", ""), project_root)
        if cfg["model"].get("checkpoint_path", "")
        else None,
        block_indices=cfg["model"].get("block_indices", [3, 5, 7, 11]),
        fpn_dim=cfg["model"].get("fpn_dim", 256),
        ppm_scales=cfg["model"].get("ppm_scales", [1, 2, 3, 6]),
        head_hidden_channels=cfg["model"].get("head_hidden_channels", 128),
        use_time_embed=cfg["model"].get("use_time_embed", True),
        use_metadata_embed=cfg["model"].get("use_metadata_embed", True),
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)

    criterion = _build_criterion(cfg, class_weights=None).to(device)

    stats = evaluate(
        model=model,
        criterion=criterion,
        loader=loader,
        device=device,
        num_classes=cfg["model"].get("num_classes", 3),
        ignore_index=cfg["data"].get("ignore_index_main", 255),
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        positive_classes=cfg["train"].get("positive_classes", [1, 2]),
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    if args.output:
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
