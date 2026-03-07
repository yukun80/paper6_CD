"""用来进行模型评估。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dataloader import build_eval_dataloader
from utils import (
    build_criterion,
    build_model,
    load_config,
    resolve_path,
    save_json,
    set_torch_home,
)
from train import run_one_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("UrbanSARFloods CH12 segmentation eval")
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--output", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config_file)

    project_root = str(Path(__file__).resolve().parents[1])
    set_torch_home(project_root)
    cfg["data"]["root"] = resolve_path(cfg["data"]["root"], project_root)

    loader = build_eval_dataloader(cfg["data"], cfg["train"], split=args.split)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.get("runtime", {}).get("require_cuda", False) and device.type != "cuda":
        raise RuntimeError("配置要求 CUDA，但当前不可用")

    model = build_model(cfg["model"]).to(device)
    criterion = build_criterion(cfg.get("loss", {}), ignore_index=int(cfg["data"].get("ignore_index", 255))).to(device)

    ckpt_path = resolve_path(args.ckpt, project_root)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)

    amp_enabled = bool(cfg.get("train", {}).get("amp", True) and device.type == "cuda")
    pos_classes = tuple(cfg.get("eval", {}).get("pos_classes", [1, 2]))
    show_progress = bool(cfg.get("train", {}).get("show_progress", True))
    progress_leave = bool(cfg.get("train", {}).get("progress_leave", False))

    with torch.no_grad():
        _, metrics = run_one_epoch(
            model=model,
            loader=loader,
            criterion=criterion,
            device=device,
            epoch=1,
            max_epochs=1,
            phase_name=f"eval-{args.split}",
            train_mode=False,
            optimizer=None,
            scaler=None,
            amp_enabled=amp_enabled,
            current_lr=0.0,
            show_progress=show_progress,
            progress_leave=progress_leave,
            ignore_index=int(cfg["data"].get("ignore_index", 255)),
            num_classes=int(cfg["model"].get("num_classes", 3)),
            aux_loss_weight=float(cfg.get("loss", {}).get("aux_loss_weight", 0.4)),
            grad_clip_norm=0.0,
            pos_classes=pos_classes,
        )

    metrics["split"] = args.split
    metrics["ckpt"] = ckpt_path

    print(metrics)

    if args.output:
        out_path = Path(args.output)
    else:
        out_dir = Path(ckpt_path).resolve().parent.parent / "eval" / args.split
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "metrics.json"
    save_json(out_path, metrics)
    print(f"评估结果已保存: {out_path}")


if __name__ == "__main__":
    main()
