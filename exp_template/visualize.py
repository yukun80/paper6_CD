"""用来进行模型预测结果可视化。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dataloader import build_eval_dataloader
from utils import (
    colorize_label,
    build_model,
    extract_logits_and_aux,
    load_config,
    make_overlay,
    resolve_path,
    set_torch_home,
    to_uint8_gray,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("UrbanSARFloods CH12 visualization")
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--max-samples", type=int, default=100)
    return parser.parse_args()


def label_to_gray(label: np.ndarray, ignore_index: int = 255) -> np.ndarray:
    """将类别标签映射成灰度图，便于快速浏览。"""

    gray = np.zeros_like(label, dtype=np.uint8)
    gray[label == 1] = 127
    gray[label == 2] = 255
    gray[label == ignore_index] = 80
    return gray


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
    ckpt_path = resolve_path(args.ckpt, project_root)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt.get("model", ckpt), strict=True)
    model.eval()

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(ckpt_path).resolve().parent.parent / "vis" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    ignore_index = int(cfg["data"].get("ignore_index", 255))
    base_channel = int(cfg.get("visualize", {}).get("base_channel", 6))
    amp_enabled = bool(cfg.get("train", {}).get("amp", True) and device.type == "cuda")

    saved = 0
    with torch.no_grad():
        for images, labels, metas in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                output = model(images)
                logits, _ = extract_logits_and_aux(output)
            preds = torch.argmax(logits, dim=1)

            bsz = preds.shape[0]
            for b in range(bsz):
                if saved >= args.max_samples:
                    break

                meta = metas[b]
                sample_id = meta["sample_id"]

                pred_np = preds[b].detach().cpu().numpy().astype(np.int64)
                gt_np = labels[b].detach().cpu().numpy().astype(np.int64)
                img_np = images[b].detach().cpu().numpy().astype(np.float32)

                c = max(0, min(base_channel, img_np.shape[0] - 1))
                base_gray = to_uint8_gray(img_np[c])

                pred_color = colorize_label(pred_np, ignore_index=ignore_index)
                gt_color = colorize_label(gt_np, ignore_index=ignore_index)
                pred_gray = label_to_gray(pred_np, ignore_index=ignore_index)
                overlay = make_overlay(base_gray, pred_color, alpha=float(cfg.get("visualize", {}).get("alpha", 0.55)))

                Image.fromarray(base_gray).save(out_dir / f"{sample_id}_base_gray.png")
                Image.fromarray(pred_gray).save(out_dir / f"{sample_id}_pred_gray.png")
                Image.fromarray(pred_color).save(out_dir / f"{sample_id}_pred_color.png")
                Image.fromarray(gt_color).save(out_dir / f"{sample_id}_gt_color.png")
                Image.fromarray(overlay).save(out_dir / f"{sample_id}_overlay.png")
                saved += 1

            if saved >= args.max_samples:
                break

    print(f"可视化完成，已保存 {saved} 个样本到: {out_dir}")


if __name__ == "__main__":
    main()
