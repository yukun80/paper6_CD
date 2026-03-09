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
    parser.add_argument("--save-ext", type=str, default="jpg", choices=["jpg", "png"])
    parser.add_argument("--jpg-quality", type=int, default=95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.save_ext == "jpg" and not (1 <= args.jpg_quality <= 100):
        raise ValueError(f"--jpg-quality 必须在 [1, 100]，当前: {args.jpg_quality}")

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
    # ch12(0-based): pre_int_vv=5, co_int_vv=7
    pre_vv_channel = int(cfg.get("visualize", {}).get("pre_vv_channel", 5))
    co_vv_channel = int(cfg.get("visualize", {}).get("co_vv_channel", 7))
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

                c_pre = max(0, min(pre_vv_channel, img_np.shape[0] - 1))
                c_co = max(0, min(co_vv_channel, img_np.shape[0] - 1))
                pre_vv_gray = to_uint8_gray(img_np[c_pre])
                co_vv_gray = to_uint8_gray(img_np[c_co])
                pred_color = colorize_label(pred_np, ignore_index=ignore_index)
                gt_color = colorize_label(gt_np, ignore_index=ignore_index)

                def _save_image(arr: np.ndarray, stem: str) -> None:
                    path = out_dir / f"{sample_id}_{stem}.{args.save_ext}"
                    if args.save_ext == "jpg":
                        Image.fromarray(arr).save(path, quality=args.jpg_quality, subsampling=0)
                    else:
                        Image.fromarray(arr).save(path)

                _save_image(pre_vv_gray, "pre_vv_gray")
                _save_image(co_vv_gray, "co_vv_gray")
                _save_image(gt_color, "gt_vis")
                _save_image(pred_color, "pred_vis")
                saved += 1

            if saved >= args.max_samples:
                break

    if args.save_ext == "jpg":
        print(
            f"可视化完成，已保存 {saved} 个样本到: {out_dir} | "
            f"输出=pre_vv_gray/co_vv_gray/gt_vis/pred_vis | 格式={args.save_ext} | quality={args.jpg_quality}"
        )
    else:
        print(
            f"可视化完成，已保存 {saved} 个样本到: {out_dir} | "
            f"输出=pre_vv_gray/co_vv_gray/gt_vis/pred_vis | 格式={args.save_ext}"
        )


if __name__ == "__main__":
    main()
