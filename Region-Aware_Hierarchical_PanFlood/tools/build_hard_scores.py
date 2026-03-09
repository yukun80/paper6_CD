#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F

_THIS = Path(__file__).resolve()
_PROJ = _THIS.parents[1]
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

from data.dataset import UrbanSARFloodsDataset
from utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser("Build hard patch scores for UrbanSARFloods training sampler")
    parser.add_argument("--config-file", required=True, type=str)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--w-urban", default=0.6, type=float)
    parser.add_argument("--w-open", default=0.2, type=float)
    parser.add_argument("--w-boundary", default=0.2, type=float)
    return parser.parse_args()


def _boundary_ratio(binary: np.ndarray) -> float:
    x = torch.from_numpy(binary.astype(np.float32))[None, None]
    edge = (F.max_pool2d(x, kernel_size=3, stride=1, padding=1) - x).clamp(min=0.0, max=1.0)
    return float(edge.mean().item())


def main():
    args = parse_args()
    cfg = load_config(args.config_file)
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

    raw_scores = []
    by_gt_name = {}
    details = []
    for _, gt_path in ds.pairs:
        with rasterio.open(gt_path) as gds:
            gt = gds.read(1).astype(np.int64)
        gt = ds._apply_label_mapping_numpy(gt)
        valid = gt != ds.ignore_index
        if valid.sum() == 0:
            open_ratio = 0.0
            urban_ratio = 0.0
            boundary = 0.0
        else:
            open_ratio = float(((gt == 1) & valid).sum() / valid.sum())
            urban_ratio = float(((gt == 2) & valid).sum() / valid.sum())
            boundary = _boundary_ratio((gt > 0) & valid)

        score = args.w_urban * urban_ratio + args.w_open * open_ratio + args.w_boundary * boundary
        raw_scores.append(score)
        name = os.path.basename(gt_path)
        by_gt_name[name] = float(score)
        details.append(
            {
                "gt_name": name,
                "open_ratio": open_ratio,
                "urban_ratio": urban_ratio,
                "boundary_ratio": boundary,
                "score_raw": float(score),
            }
        )

    arr = np.asarray(raw_scores, dtype=np.float64)
    if arr.size > 0:
        lo = float(np.percentile(arr, 5))
        hi = float(np.percentile(arr, 95))
        norm = np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    else:
        norm = arr

    by_gt_name_norm = {}
    for i, d in enumerate(details):
        d["score"] = float(norm[i]) if norm.size > 0 else 0.0
        by_gt_name_norm[d["gt_name"]] = d["score"]

    top_idx = np.argsort(-norm)[: min(30, len(norm))] if norm.size > 0 else np.array([], dtype=np.int64)
    hardest = [details[int(i)] for i in top_idx.tolist()]

    out = {
        "split": args.split,
        "weights": {"urban": args.w_urban, "open": args.w_open, "boundary": args.w_boundary},
        "num_samples": len(details),
        "by_gt_name": by_gt_name_norm,
        "scores": [float(x) for x in norm.tolist()],
        "hardest_topk": hardest,
    }

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps({"saved": args.output, "num_samples": len(details)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
