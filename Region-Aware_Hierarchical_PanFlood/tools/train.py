#!/usr/bin/env python3
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch

_THIS = Path(__file__).resolve()
_PROJ = _THIS.parents[1]
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.config import dump_config, ensure_dir, load_config
from utils.logger import JsonlLogger, setup_logger
from utils.model_utils import count_parameters, summarize_module_parameters
from utils.runtime import maybe_disable_xformers, resolve_device
from utils.seed import set_seed

"""
python Region-Aware_Hierarchical_PanFlood/tools/train.py \
    --config-file Region-Aware_Hierarchical_PanFlood/config/stage4_full_prompt_refine.yaml \
    --device cuda
"""


def parse_args():
    parser = argparse.ArgumentParser("Region-Aware Hierarchical PanFlood training")
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--max-steps-per-epoch", type=int, default=-1)
    parser.add_argument("--max-eval-steps", type=int, default=-1)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config_file)
    set_seed(int(cfg.get("seed", 42)))

    root = str(_PROJ.parent)
    out_root = args.output_dir or os.path.join(root, cfg["train"]["output_dir"])
    run_name = cfg["train"].get("run_name", datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    work_dir = os.path.join(out_root, run_name)
    ensure_dir(work_dir)

    logger = setup_logger(work_dir)
    jlogger = JsonlLogger(os.path.join(work_dir, "metrics.jsonl"))
    dump_config(cfg, os.path.join(work_dir, "resolved_config.yaml"))

    runtime_cfg = cfg.get("runtime", {})
    device, device_report = resolve_device(
        requested=args.device,
        require_cuda=bool(runtime_cfg.get("require_cuda", False)),
    )
    maybe_disable_xformers(device.type == "cpu")
    from engine.builder import build_datasets, build_loaders, build_loss, build_model, build_optimizer
    from engine.trainer import Trainer, is_better, resolve_amp_dtype

    with open(os.path.join(work_dir, "device_report.json"), "w", encoding="utf-8") as f:
        json.dump(device_report, f, indent=2, ensure_ascii=False)
    logger.info(f"Using device: {device.type}, report={device_report}")

    train_ds, val_ds = build_datasets(cfg, project_root=root)
    train_loader, val_loader = build_loaders(cfg, train_ds, val_ds, project_root=root)
    with open(os.path.join(work_dir, "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "train": train_ds.get_label_mapping_report(),
                "val": val_ds.get_label_mapping_report(),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    model = build_model(cfg).to(device)
    with open(os.path.join(work_dir, "module_param_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summarize_module_parameters(model), f, indent=2, ensure_ascii=False)
    criterion = build_loss(cfg)

    # 分阶段训练：先冻结主干再解冻最后 N 个 block。
    freeze_epochs = int(cfg["train"].get("freeze_backbone_epochs", 0))
    unfreeze_last_n = int(cfg["train"].get("unfreeze_last_n_blocks", 4))
    if freeze_epochs > 0:
        model.backbone.freeze_backbone_keep_adapters()

    optimizer, scheduler = build_optimizer(cfg, model)

    amp_cfg = cfg.get("runtime", {}).get("amp", {})
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_classes=3,
        ignore_index=int(cfg["data"].get("ignore_index", 255)),
        class_names=cfg["data"].get("class_names", ["non-flood", "flood-open", "flood-urban"]),
        amp_enabled=bool(amp_cfg.get("enabled", True)),
        amp_dtype=resolve_amp_dtype(str(amp_cfg.get("dtype", "fp16"))),
        grad_clip_norm=float(cfg["train"].get("grad_clip_norm", 0.0)),
    )

    params = count_parameters(model)
    logger.info(f"Model params: total={params['total']}, trainable={params['trainable']}")
    logger.info(f"Backbone load msg: {model.backbone.load_msg}")

    start_epoch = 0
    best_metric = None
    best_mode = cfg["train"].get("best_mode", "max")
    best_key = cfg["train"].get("best_metric", "val_mIoU")
    if args.resume:
        state = load_checkpoint(args.resume, model, optimizer, scheduler, map_location="cpu")
        start_epoch = state["epoch"] + 1
        best_metric = state["best_metric"]
        logger.info(f"Resumed from {args.resume}, start_epoch={start_epoch}, best={best_metric}")

    epochs = int(cfg["train"]["epochs"])
    log_interval = int(cfg["train"].get("log_interval", 50))

    for epoch in range(start_epoch, epochs):
        if epoch == freeze_epochs and freeze_epochs > 0:
            model.backbone.unfreeze_last_n_blocks(unfreeze_last_n)
            logger.info(f"[epoch={epoch}] unfreeze backbone last {unfreeze_last_n} blocks")

        train_stats = trainer.train_one_epoch(
            train_loader,
            epoch=epoch,
            logger=logger,
            log_interval=log_interval,
            max_steps=args.max_steps_per_epoch,
        )
        val_stats = trainer.evaluate(val_loader, split="val", max_steps=args.max_eval_steps)

        merged = {"epoch": epoch, **train_stats, **val_stats}
        jlogger.log(merged, step=epoch)

        logger.info(
            f"[epoch={epoch}] train_loss={train_stats['train_loss_total']:.4f} "
            f"val_mIoU={val_stats['val_mIoU']:.4f} val_open_IoU={val_stats['val_open_IoU']:.4f} "
            f"val_urban_IoU={val_stats['val_urban_IoU']:.4f}"
        )

        score = float(merged[best_key])
        improved = is_better(score, best_metric, mode=best_mode)
        if improved:
            best_metric = score

        latest, best = save_checkpoint(
            output_dir=work_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_metric=best_metric if best_metric is not None else score,
            metric_name=best_key,
            is_best=improved,
        )
        logger.info(f"Saved latest checkpoint: {latest}")
        if best is not None:
            logger.info(f"Saved best checkpoint: {best}")

    with open(os.path.join(work_dir, "train_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"best_metric": best_metric, "best_key": best_key}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
