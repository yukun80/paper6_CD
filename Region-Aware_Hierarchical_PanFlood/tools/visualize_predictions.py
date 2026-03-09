#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

_THIS = Path(__file__).resolve()
_PROJ = _THIS.parents[1]
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

from data.dataset import UrbanSARFloodsDataset, collate_fn
from utils.checkpoint import load_checkpoint
from utils.config import load_config
from utils.runtime import maybe_disable_xformers, resolve_device


try:
    import matplotlib.pyplot as plt
except Exception as e:  # pragma: no cover
    raise RuntimeError("matplotlib is required for visualization") from e


def parse_args():
    parser = argparse.ArgumentParser("Visualize predictions")
    parser.add_argument("--config-file", required=True, type=str)
    parser.add_argument("--ckpt", required=True, type=str)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--out-dir", required=True, type=str)
    parser.add_argument("--max-samples", default=20, type=int)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


def _norm01(x):
    x = np.asarray(x)
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    return np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)


def _rgb(mask):
    cmap = np.array([[20, 20, 20], [40, 130, 255], [255, 60, 80]], dtype=np.uint8)
    return cmap[np.clip(mask, 0, 2)]


def main():
    args = parse_args()
    cfg = load_config(args.config_file)
    os.makedirs(args.out_dir, exist_ok=True)

    root = str(_PROJ.parent)
    data_cfg = cfg["data"]
    split_map = {
        "train": data_cfg["train_split"],
        "val": data_cfg.get("val_split", data_cfg["train_split"]),
        "test": data_cfg.get("test_split", data_cfg.get("val_split", data_cfg["train_split"])),
    }

    ds = UrbanSARFloodsDataset(
        data_root=os.path.normpath(os.path.join(root, data_cfg["root"])),
        split_file=split_map[args.split],
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

    runtime_cfg = cfg.get("runtime", {})
    device, _ = resolve_device(
        requested=args.device,
        require_cuda=bool(runtime_cfg.get("require_cuda", False)),
    )
    maybe_disable_xformers(device.type == "cpu")
    from engine.builder import build_model

    model = build_model(cfg).to(device)
    load_checkpoint(args.ckpt, model, map_location="cpu")
    model.eval()

    device = next(model.parameters()).device
    for i in range(min(args.max_samples, len(ds))):
        x_dict, y, meta = ds[i]
        x = {k: v.unsqueeze(0).to(device) for k, v in x_dict.items()}
        with torch.no_grad():
            out = model(x)

        pred = out["final_logits"].argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        gt = y.numpy().astype(np.uint8)

        # 选取前4个输入通道做展示
        img = x_dict["imgs"].numpy()
        ch0 = _norm01(img[0])
        ch1 = _norm01(img[min(1, img.shape[0] - 1)])
        ch2 = _norm01(img[min(2, img.shape[0] - 1)])
        ch3 = _norm01(img[min(3, img.shape[0] - 1)])

        floodness = torch.sigmoid(out["floodness_logits"])[0, 0].cpu().numpy()
        open_p = torch.sigmoid(out["open_logits"])[0, 0].cpu().numpy()
        urban_p = torch.sigmoid(out["urban_logits"])[0, 0].cpu().numpy()

        fig, axes = plt.subplots(2, 5, figsize=(16, 7))
        axes = axes.flatten()
        axes[0].imshow(ch0, cmap="gray")
        axes[0].set_title("input ch0")
        axes[1].imshow(ch1, cmap="gray")
        axes[1].set_title("input ch1")
        axes[2].imshow(ch2, cmap="gray")
        axes[2].set_title("input ch2")
        axes[3].imshow(ch3, cmap="gray")
        axes[3].set_title("input ch3")
        axes[4].imshow(_rgb(gt))
        axes[4].set_title("GT")

        axes[5].imshow(floodness, cmap="viridis")
        axes[5].set_title("floodness")
        axes[6].imshow(open_p, cmap="Blues")
        axes[6].set_title("open expert")
        axes[7].imshow(urban_p, cmap="Reds")
        axes[7].set_title("urban expert")
        axes[8].imshow(_rgb(pred))
        axes[8].set_title("final pred")
        axes[9].imshow(torch.softmax(out["router_logits"], dim=1)[0, 2].cpu().numpy(), cmap="magma")
        axes[9].set_title("router ambiguous")

        for ax in axes:
            ax.axis("off")

        fig.suptitle(Path(meta["sar_path"]).name)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"viz_{i:04d}.png"), dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
