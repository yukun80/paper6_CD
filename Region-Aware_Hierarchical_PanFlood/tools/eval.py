#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_THIS = Path(__file__).resolve()
_PROJ = _THIS.parents[1]
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

from data.dataset import UrbanSARFloodsDataset, collate_fn
from utils.checkpoint import load_checkpoint
from utils.config import load_config
from utils.runtime import maybe_disable_xformers, resolve_device


def parse_args():
    parser = argparse.ArgumentParser("Region-Aware Hierarchical PanFlood evaluation")
    parser.add_argument("--config-file", required=True, type=str)
    parser.add_argument("--ckpt", required=True, type=str)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--output", default="", type=str)
    parser.add_argument("--max-steps", default=-1, type=int)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config_file)

    root = str(_PROJ.parent)
    data_cfg = cfg["data"]
    split_file = data_cfg.get("val_split", "Valid_dataset.txt")
    if args.split == "test":
        # 当前版本默认将 test 入口映射到 valid split。
        split_file = data_cfg.get("test_split", data_cfg.get("val_split", "Valid_dataset.txt"))

    ds = UrbanSARFloodsDataset(
        data_root=os.path.normpath(os.path.join(root, data_cfg["root"])),
        split_file=split_file,
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
    loader = DataLoader(
        ds,
        batch_size=cfg["train"].get("val_batch_size", 1),
        shuffle=False,
        num_workers=cfg["train"].get("num_workers", 4),
        pin_memory=True,
        collate_fn=collate_fn,
    )

    runtime_cfg = cfg.get("runtime", {})
    device, device_report = resolve_device(
        requested=args.device,
        require_cuda=bool(runtime_cfg.get("require_cuda", False)),
    )
    maybe_disable_xformers(device.type == "cpu")
    from engine.builder import build_loss, build_model
    from engine.trainer import Trainer, resolve_amp_dtype

    print(json.dumps({"device": str(device), "device_report": device_report}, ensure_ascii=False, indent=2))
    model = build_model(cfg).to(device)
    criterion = build_loss(cfg)
    load_checkpoint(args.ckpt, model, map_location="cpu")

    amp_cfg = cfg.get("runtime", {}).get("amp", {})
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-5),
        scheduler=None,
        device=device,
        num_classes=3,
        ignore_index=int(data_cfg.get("ignore_index", 255)),
        class_names=data_cfg.get("class_names", ["non-flood", "flood-open", "flood-urban"]),
        amp_enabled=bool(amp_cfg.get("enabled", True)),
        amp_dtype=resolve_amp_dtype(str(amp_cfg.get("dtype", "fp16"))),
    )

    metrics = trainer.evaluate(loader, split=args.split, max_steps=args.max_steps)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
