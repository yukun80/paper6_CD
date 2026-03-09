#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
from PIL import Image

_THIS = Path(__file__).resolve()
_PROJ = _THIS.parents[1]
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

from data.channel_specs import build_channel_id_arrays, build_channel_layout
from data.dataset import _compute_engineered_from_8ch
from utils.checkpoint import load_checkpoint
from utils.config import load_config
from utils.runtime import maybe_disable_xformers, resolve_device


def parse_args():
    parser = argparse.ArgumentParser("Region-Aware Hierarchical PanFlood inference")
    parser.add_argument("--config-file", required=True, type=str)
    parser.add_argument("--ckpt", required=True, type=str)
    parser.add_argument("--sar-path", type=str, default="")
    parser.add_argument("--sar-dir", type=str, default="")
    parser.add_argument("--out-dir", required=True, type=str)
    parser.add_argument("--device", default="auto", type=str, choices=["auto", "cuda", "cpu"])
    parser.add_argument("--max-samples", default=50, type=int)
    return parser.parse_args()


def _to_color(mask: np.ndarray) -> np.ndarray:
    cmap = np.array(
        [
            [20, 20, 20],
            [40, 130, 255],
            [255, 60, 80],
        ],
        dtype=np.uint8,
    )
    return cmap[np.clip(mask, 0, 2)]


def _build_x_dict(sar_path: str, input_mode: str, mean, std, crop_size: int):
    with rasterio.open(sar_path) as ds:
        arr = ds.read().astype(np.float32)

    x = torch.from_numpy(arr)
    if input_mode == "8ch_plus_engineered":
        if x.shape[0] != 8:
            raise ValueError(f"engineered mode expects 8ch input, got {x.shape[0]} @ {sar_path}")
        x = torch.cat([x, _compute_engineered_from_8ch(x)], dim=0)
    elif input_mode == "8ch" and x.shape[0] != 8:
        raise ValueError(f"8ch mode expects 8 channels, got {x.shape[0]}")
    elif input_mode == "12ch" and x.shape[0] != 12:
        raise ValueError(f"12ch mode expects 12 channels, got {x.shape[0]}")

    finite = torch.isfinite(x)
    m = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
    s = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1).clamp(min=1e-6)
    x = torch.where(finite, x, m.expand_as(x))
    x = (x - m) / s

    # 与训练保持一致，默认中心裁剪到可被 patch size(14) 整除尺寸（如252）。
    h, w = x.shape[-2:]
    ch = cw = int(crop_size)
    if ch <= h and cw <= w:
        top = (h - ch) // 2
        left = (w - cw) // 2
        x = x[:, top : top + ch, left : left + cw]

    layout = build_channel_layout(input_mode)
    role_ids = build_channel_id_arrays(layout)
    c = x.shape[0]

    x_dict = {
        "imgs": x.unsqueeze(0),
        "chn_ids": torch.tensor(role_ids["chn_ids"][:c], dtype=torch.long).unsqueeze(0),
        "time_ids": torch.tensor(role_ids["time_ids"][:c], dtype=torch.long).unsqueeze(0),
        "feature_type_ids": torch.tensor(role_ids["feature_type_ids"][:c], dtype=torch.long).unsqueeze(0),
        "temporal_role_ids": torch.tensor(role_ids["temporal_role_ids"][:c], dtype=torch.long).unsqueeze(0),
        "polarization_ids": torch.tensor(role_ids["polarization_ids"][:c], dtype=torch.long).unsqueeze(0),
        "source_role_ids": torch.tensor(role_ids["source_role_ids"][:c], dtype=torch.long).unsqueeze(0),
    }
    return x_dict


def main():
    args = parse_args()
    cfg = load_config(args.config_file)
    os.makedirs(args.out_dir, exist_ok=True)

    runtime_cfg = cfg.get("runtime", {})
    device, device_report = resolve_device(
        requested=args.device,
        require_cuda=bool(runtime_cfg.get("require_cuda", False)),
    )
    maybe_disable_xformers(device.type == "cpu")
    from engine.builder import build_model

    print(f"device={device}, cuda_available={device_report['cuda_available']}, torch_cuda={device_report['torch_cuda_compiled']}")

    model = build_model(cfg).to(device)
    load_checkpoint(args.ckpt, model, map_location="cpu")
    model.eval()

    data_cfg = cfg["data"]
    input_mode = data_cfg["input_mode"]
    mean = data_cfg["mean"]
    std = data_cfg["std"]
    crop_size = int(data_cfg.get("crop_size", 252))

    sar_paths = []
    if args.sar_path:
        sar_paths = [args.sar_path]
    elif args.sar_dir:
        sar_paths = sorted(str(x) for x in Path(args.sar_dir).glob("*_SAR.tif"))[: args.max_samples]
    else:
        raise ValueError("Please provide --sar-path or --sar-dir")

    with torch.no_grad():
        for i, sar_path in enumerate(sar_paths):
            x_dict = _build_x_dict(sar_path, input_mode, mean, std, crop_size=crop_size)
            x_dict = {k: v.to(device) for k, v in x_dict.items()}
            out = model(x_dict)
            pred = out["final_logits"].argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            color = _to_color(pred)

            stem = Path(sar_path).stem.replace("_SAR", "")
            Image.fromarray(pred).save(os.path.join(args.out_dir, f"{stem}_pred.png"))
            Image.fromarray(color).save(os.path.join(args.out_dir, f"{stem}_pred_color.png"))

            if i % 10 == 0:
                print(f"[{i+1}/{len(sar_paths)}] {sar_path}")


if __name__ == "__main__":
    main()
