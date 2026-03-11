"""监督参考实验推理入口。"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image, ImageDraw

from .metrics import binary_dice, binary_iou
from .optimizer import PromptOptimizationEnv, rule_greedy_optimize
from .reference_config import load_reference_config
from .reference_data import GF3TargetTileDataset, UrbanSARReferenceTileDataset, ensure_reference_splits, select_reference_id
from .reference_models import ReferencePromptProposalNet
from .reference_policy import PolicyActionScorer, infer_policy_feature_dim, policy_greedy_optimize
from .segmenter import build_segmenter
from .supervised_prompts import ReferencePromptCandidateGenerator


def parse_args():
    parser = argparse.ArgumentParser("Run reference-guided prompt pipeline")
    parser.add_argument("--config-file", type=str, default="sar_prompt_flood/config/urban_sar_reference.json")
    parser.add_argument("--proposal-ckpt", type=str, default="")
    parser.add_argument("--policy-ckpt", type=str, default="")
    parser.add_argument("--split", type=str, default="")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_proposal_model(cfg, ckpt_path: str, device: str) -> ReferencePromptProposalNet:
    model = ReferencePromptProposalNet(**cfg["model"]).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def load_policy_model(cfg, ckpt_path: str, device: str) -> PolicyActionScorer:
    model = PolicyActionScorer(
        infer_policy_feature_dim(),
        hidden_dim=int(cfg["policy"].get("hidden_dim", 128)),
        dropout=float(cfg["policy"].get("dropout", 0.1)),
    ).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    cfg = load_reference_config(args.config_file)
    ref_cfg = cfg["reference_data"]
    target_cfg = cfg["target_data"]
    seed = int(cfg["runtime"].get("seed", ref_cfg.get("seed", 42)))
    set_seed(seed)
    ensure_reference_splits(ref_cfg["root"], ref_cfg["split_dir"], seed=seed, val_ratio=ref_cfg.get("val_ratio", 0.2))

    device = cfg["train"].get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    proposal_ckpt = args.proposal_ckpt or str(Path(cfg["train"]["work_dir"]) / "best_reference_prompt.pth")
    policy_ckpt = args.policy_ckpt or str(Path(cfg["train"]["work_dir"]) / "best_prompt_policy.pth")

    proposal_model = load_proposal_model(cfg, proposal_ckpt, device)
    generator = ReferencePromptCandidateGenerator(proposal_model, device=device, **cfg["prompts"])
    use_policy = bool(cfg["inference"].get("use_policy", True)) and Path(policy_ckpt).exists()
    policy_model = load_policy_model(cfg, policy_ckpt, device) if use_policy else None
    segmenter = build_segmenter(cfg)

    query_ds = GF3TargetTileDataset(
        root=target_cfg["root"],
        max_samples=min(
            int(cfg["inference"].get("max_samples", -1)),
            int(target_cfg.get("max_samples", -1)),
        ) if int(cfg["inference"].get("max_samples", -1)) > 0 and int(target_cfg.get("max_samples", -1)) > 0 else max(
            int(cfg["inference"].get("max_samples", -1)),
            int(target_cfg.get("max_samples", -1)),
        ),
    )
    ref_ds = UrbanSARReferenceTileDataset(
        root=ref_cfg["root"],
        split=ref_cfg.get("ref_bank_split", "ref_bank.txt"),
        split_dir=ref_cfg["split_dir"],
        seed=seed,
        val_ratio=ref_cfg.get("val_ratio", 0.2),
        ignore_index=ref_cfg.get("ignore_index", 255),
        max_samples=ref_cfg.get("max_ref_samples", -1),
    )
    ref_ids = [sid for sid in ref_ds.sample_ids if ref_ds.get_positive_ratio(sid) > 0.0] or ref_ds.sample_ids

    out_dir = Path(cfg["inference"]["output_dir"]).resolve()
    pred_dir = out_dir / "predictions"
    vis_dir = out_dir / "visuals"
    pred_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict] = []
    reference_counter: Counter[str] = Counter()
    ignore_index = int(ref_cfg.get("ignore_index", 255))

    for query in query_ds:
        ref_id = select_reference_id(query, ref_ds, ref_ids, exclude_id=None, seed=seed)
        reference_counter.update([ref_id])
        reference = ref_ds.get_sample_by_id(ref_id)
        prompt_set = generator.generate(query, reference)
        env = PromptOptimizationEnv(
            prompt_set=prompt_set,
            segmenter=segmenter,
            max_steps=cfg["optimizer"].get("max_steps", 8),
            min_positive_points=cfg["optimizer"].get("min_positive_points", 2),
            max_positive_points=cfg["optimizer"].get("max_positive_points", 10),
            min_negative_points=cfg["optimizer"].get("min_negative_points", 2),
            max_negative_points=cfg["optimizer"].get("max_negative_points", 10),
        )
        init_summary = env.export_summary()
        if use_policy and policy_model is not None:
            opt_summary = policy_greedy_optimize(env, policy_model, device=device, stop_threshold=float(cfg["policy"].get("stop_threshold", 0.0)))
        else:
            opt_summary = rule_greedy_optimize(env)
        record = {
            "sample_id": query["sample_id"],
            "reference_id": ref_id,
            "actions": opt_summary.action_history,
            "num_pos": len(opt_summary.pos_points),
            "num_neg": len(opt_summary.neg_points),
            "low_confidence": bool(prompt_set.low_confidence),
            "init_mask_area_ratio": float(init_summary.mask.mean()),
            "opt_mask_area_ratio": float(opt_summary.mask.mean()),
        }
        if "gt" in query:
            gt = query["gt"]
            record.update(
                {
                    "init_iou": binary_iou(torch.from_numpy(init_summary.mask), gt, ignore_index=ignore_index),
                    "opt_iou": binary_iou(torch.from_numpy(opt_summary.mask), gt, ignore_index=ignore_index),
                    "init_dice": binary_dice(torch.from_numpy(init_summary.mask), gt, ignore_index=ignore_index),
                    "opt_dice": binary_dice(torch.from_numpy(opt_summary.mask), gt, ignore_index=ignore_index),
                }
            )
        records.append(record)
        save_mask_png(pred_dir / f"{query['sample_id']}_mask.png", opt_summary.mask)
        if cfg["inference"].get("save_geotiff", True):
            save_mask_geotiff(pred_dir / f"{query['sample_id']}_mask.tif", opt_summary.mask, query["meta"]["profile"])
        if cfg["inference"].get("save_visuals", True):
            save_visual(vis_dir / f"{query['sample_id']}.png", query, init_summary, opt_summary)

    summary = {
        "num_samples": len(records),
        "used_policy": use_policy,
        "mean_init_mask_area_ratio": float(np.mean([x["init_mask_area_ratio"] for x in records])) if records else 0.0,
        "mean_opt_mask_area_ratio": float(np.mean([x["opt_mask_area_ratio"] for x in records])) if records else 0.0,
        "mean_num_pos": float(np.mean([x["num_pos"] for x in records])) if records else 0.0,
        "mean_num_neg": float(np.mean([x["num_neg"] for x in records])) if records else 0.0,
        "low_confidence_tiles": int(sum(1 for x in records if x["low_confidence"])),
        "reference_selection_top10": reference_counter.most_common(10),
    }
    if records and "opt_iou" in records[0]:
        summary.update(
            {
                "mean_init_iou": float(np.mean([x["init_iou"] for x in records])),
                "mean_opt_iou": float(np.mean([x["opt_iou"] for x in records])),
                "mean_init_dice": float(np.mean([x["init_dice"] for x in records])),
                "mean_opt_dice": float(np.mean([x["opt_dice"] for x in records])),
            }
        )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "tile_metrics.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def save_mask_png(path: Path, mask: np.ndarray) -> None:
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def save_mask_geotiff(path: Path, mask: np.ndarray, profile: Dict) -> None:
    import rasterio

    out_profile = profile.copy()
    out_profile.update(
        dtype="uint8",
        count=1,
        compress="deflate",
        predictor=1,
        nodata=0,
        height=mask.shape[0],
        width=mask.shape[1],
    )
    with rasterio.open(path, "w", **out_profile) as ds:
        ds.write(mask.astype(np.uint8), 1)


def save_visual(path: Path, query: Dict, init_summary, opt_summary) -> None:
    image = Image.fromarray(query["pseudo_rgb"].numpy().astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    for x, y in init_summary.pos_points:
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(0, 255, 0))
    for x, y in init_summary.neg_points:
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(255, 0, 255))
    for x, y in opt_summary.pos_points:
        draw.rectangle((x - 2, y - 2, x + 2, y + 2), outline=(255, 255, 0))
    image.save(path)


if __name__ == "__main__":
    main()
