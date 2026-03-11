"""训练监督式 prompt 动作评分器。"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from .metrics import binary_iou
from .optimizer import PromptOptimizationEnv
from .reference_config import load_reference_config
from .reference_data import UrbanSARReferenceTileDataset, ensure_reference_splits, select_reference_id
from .reference_models import ReferencePromptProposalNet
from .reference_policy import (
    PolicyActionScorer,
    collect_supervised_action_samples,
    infer_policy_feature_dim,
    policy_greedy_optimize,
)
from .segmenter import build_segmenter
from .supervised_prompts import ReferencePromptCandidateGenerator


def parse_args():
    parser = argparse.ArgumentParser("Train prompt policy scorer")
    parser.add_argument("--config-file", type=str, default="sar_prompt_flood/config/urban_sar_reference.json")
    parser.add_argument("--proposal-ckpt", type=str, default="")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_proposal_model(cfg, ckpt_path: str, device: str) -> ReferencePromptProposalNet:
    model = ReferencePromptProposalNet(**cfg["model"]).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    cfg = load_reference_config(args.config_file)
    data_cfg = cfg["reference_data"]
    seed = int(cfg["runtime"].get("seed", data_cfg.get("seed", 42)))
    set_seed(seed)
    ensure_reference_splits(data_cfg["root"], data_cfg["split_dir"], seed=seed, val_ratio=data_cfg.get("val_ratio", 0.2))

    device = cfg["train"].get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    proposal_ckpt = args.proposal_ckpt or str(Path(cfg["train"]["work_dir"]) / "best_reference_prompt.pth")
    proposal_model = load_proposal_model(cfg, proposal_ckpt, device)
    generator = ReferencePromptCandidateGenerator(proposal_model, device=device, **cfg["prompts"])
    segmenter = build_segmenter(cfg)

    train_ds = UrbanSARReferenceTileDataset(
        root=data_cfg["root"],
        split=data_cfg.get("train_split", "train.txt"),
        split_dir=data_cfg["split_dir"],
        seed=seed,
        val_ratio=data_cfg.get("val_ratio", 0.2),
        ignore_index=data_cfg.get("ignore_index", 255),
        max_samples=data_cfg.get("max_train_samples", -1),
    )
    val_ds = UrbanSARReferenceTileDataset(
        root=data_cfg["root"],
        split=data_cfg.get("val_split", "val.txt"),
        split_dir=data_cfg.get("split_dir"),
        seed=seed,
        val_ratio=data_cfg.get("val_ratio", 0.2),
        ignore_index=data_cfg.get("ignore_index", 255),
        max_samples=data_cfg.get("max_val_samples", -1),
    )
    ref_ds = UrbanSARReferenceTileDataset(
        root=data_cfg["root"],
        split=data_cfg.get("ref_bank_split", "ref_bank.txt"),
        split_dir=data_cfg.get("split_dir"),
        seed=seed,
        val_ratio=data_cfg.get("val_ratio", 0.2),
        ignore_index=data_cfg.get("ignore_index", 255),
        max_samples=data_cfg.get("max_ref_samples", -1),
    )
    ref_ids = [sid for sid in ref_ds.sample_ids if ref_ds.get_positive_ratio(sid) > 0.0] or ref_ds.sample_ids

    scorer = PolicyActionScorer(infer_policy_feature_dim(), hidden_dim=int(cfg["policy"].get("hidden_dim", 128)), dropout=float(cfg["policy"].get("dropout", 0.1))).to(device)
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=float(cfg["train"].get("lr", 1e-3)), weight_decay=float(cfg["train"].get("weight_decay", 1e-4)))
    work_dir = Path(cfg["train"]["work_dir"]).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    epochs = int(cfg["train"].get("epochs", 8))
    ignore_index = int(data_cfg.get("ignore_index", 255))
    history: List[dict] = []
    best_val_gain = -1e9

    for epoch in range(1, epochs + 1):
        scorer.train()
        epoch_loss = 0.0
        num_samples = 0
        for query in train_ds:
            ref_id = select_reference_id(query, ref_ds, ref_ids, exclude_id=query["sample_id"], seed=seed + epoch)
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
                objective_weights=cfg["optimizer"].get("objective_weights"),
            )
            action_samples = collect_supervised_action_samples(env, query["gt"].numpy(), ignore_index=ignore_index)
            if not action_samples:
                continue
            x = torch.from_numpy(np.stack([item.feature for item in action_samples], axis=0)).to(device)
            y = torch.tensor([item.target for item in action_samples], dtype=torch.float32, device=device)
            optimizer.zero_grad()
            pred = scorer(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            num_samples += 1

        scorer.eval()
        val_gains = []
        with torch.no_grad():
            for query in val_ds:
                ref_id = select_reference_id(query, ref_ds, ref_ids, exclude_id=query["sample_id"], seed=seed + epoch + 99)
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
                    objective_weights=cfg["optimizer"].get("objective_weights"),
                )
                init_summary = env.export_summary()
                summary = policy_greedy_optimize(env, scorer, device=device, stop_threshold=float(cfg["policy"].get("stop_threshold", 0.0)))
                init_iou = binary_iou(torch.from_numpy(init_summary.mask), query["gt"], ignore_index=ignore_index)
                opt_iou = binary_iou(torch.from_numpy(summary.mask), query["gt"], ignore_index=ignore_index)
                val_gains.append(float(opt_iou - init_iou))
        mean_train_loss = epoch_loss / max(num_samples, 1)
        mean_val_gain = float(np.mean(val_gains)) if val_gains else 0.0
        history.append({"epoch": epoch, "train_loss": mean_train_loss, "val_gain": mean_val_gain})
        print(json.dumps(history[-1], ensure_ascii=False))
        torch.save({"model": scorer.state_dict(), "cfg": cfg, "epoch": epoch}, work_dir / "last_prompt_policy.pth")
        if mean_val_gain > best_val_gain:
            best_val_gain = mean_val_gain
            torch.save({"model": scorer.state_dict(), "cfg": cfg, "epoch": epoch, "best_val_gain": best_val_gain}, work_dir / "best_prompt_policy.pth")

    (work_dir / "prompt_policy_history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
