import argparse
import json
import os
from typing import Dict

import torch
import yaml
from torch.utils.data import DataLoader

from dinov2.layers.attention import XFORMERS_AVAILABLE, XFORMERS_ENABLED
from urban_floods.dataset import UrbanSARFloodsSegDataset, collate_urban_floods
from urban_floods.model import PanopticonUrbanSeg
from urban_floods.train import _resolve_runtime, evaluate


def parse_args():
    """解析评估脚本命令行参数。"""
    parser = argparse.ArgumentParser("Panopticon UrbanSARFloods evaluation")
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["val", "train"])
    parser.add_argument("--output", type=str, default="")
    return parser.parse_args()


def load_cfg(path: str) -> Dict:
    """加载 YAML 配置。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path: str, base_dir: str) -> str:
    """将相对路径解析到项目根目录。"""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def assert_cuda_runtime(cfg: Dict):
    """在评估前强校验 CUDA 与 xformers 环境。"""
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


def main():
    """评估入口：加载数据、权重并输出指标。"""
    args = parse_args()
    cfg = load_cfg(args.config_file)
    assert_cuda_runtime(cfg)
    project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

    runtime_cfg = cfg.get("runtime", {})
    if bool(runtime_cfg.get("allow_tf32", True)):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    amp_enabled, amp_dtype, _ = _resolve_runtime(cfg)

    # 构建评估数据集（默认不做随机增强）。
    data_cfg = cfg["data"]
    split_file = data_cfg["val_split"] if args.split == "val" else data_cfg["train_split"]
    ds = UrbanSARFloodsSegDataset(
        data_root=resolve_path(data_cfg["root"], project_root),
        split_file=split_file,
        channel_ids=data_cfg["channel_ids"],
        time_ids=data_cfg["time_ids"],
        mean=data_cfg["mean"],
        std=data_cfg["std"],
        ignore_index=data_cfg.get("ignore_index", 255),
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
        collate_fn=collate_urban_floods,
        drop_last=False,
    )

    device = torch.device("cuda")
    print(
        f"[runtime] device={device}, xformers_enabled={XFORMERS_ENABLED and XFORMERS_AVAILABLE}, "
        f"amp_enabled={amp_enabled}, amp_dtype={amp_dtype}"
    )
    # 先用配置中的预训练权重构图，再覆盖为指定 ckpt 的训练后权重。
    model = PanopticonUrbanSeg(
        ckpt_path=resolve_path(cfg["model"]["checkpoint_path"], project_root),
        num_classes=cfg["model"]["num_classes"],
        block_indices=cfg["model"]["block_indices"],
        decode_channels=cfg["model"]["decode_channels"],
        use_time_embed=cfg["model"].get("use_time_embed", True),
    ).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)

    # 复用 train.py 中的 evaluate 逻辑，确保训练/评估指标定义一致。
    stats = evaluate(
        model=model,
        loader=loader,
        device=device,
        num_classes=cfg["model"]["num_classes"],
        ignore_index=cfg["data"].get("ignore_index", 255),
        loss_cfg=cfg["loss"],
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
