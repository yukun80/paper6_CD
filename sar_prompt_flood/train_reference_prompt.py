"""训练参考引导 prompt proposal。"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from .metrics import binary_iou
from .reference_config import load_reference_config
from .reference_data import ReferenceQueryPairDataset, UrbanSARReferenceTileDataset, ensure_reference_splits
from .reference_models import ReferencePromptProposalNet, proposal_loss

"""
python -m sar_prompt_flood.train_reference_prompt \
    --config-file sar_prompt_flood/config/urban_sar_reference.json
"""


def parse_args():
    parser = argparse.ArgumentParser("Train reference-guided prompt proposal")
    parser.add_argument("--config-file", type=str, default="sar_prompt_flood/config/urban_sar_reference.json")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_pair_batch(batch: List[Dict]) -> Dict[str, Dict[str, torch.Tensor]]:
    out = {"query": {}, "reference": {}}
    for side in ["query", "reference"]:
        sample0 = batch[0][side]
        for key, value in sample0.items():
            if isinstance(value, torch.Tensor):
                out[side][key] = torch.stack([item[side][key] for item in batch], dim=0)
            else:
                out[side][key] = [item[side][key] for item in batch]
    return out


def main() -> None:
    args = parse_args()
    cfg = load_reference_config(args.config_file)
    data_cfg = cfg["reference_data"]
    seed = int(cfg["runtime"].get("seed", data_cfg.get("seed", 42)))
    set_seed(seed)
    ensure_reference_splits(data_cfg["root"], data_cfg["split_dir"], seed=seed, val_ratio=data_cfg.get("val_ratio", 0.2))

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
        split_dir=data_cfg["split_dir"],
        seed=seed,
        val_ratio=data_cfg.get("val_ratio", 0.2),
        ignore_index=data_cfg.get("ignore_index", 255),
        max_samples=data_cfg.get("max_val_samples", -1),
    )
    ref_ds = UrbanSARReferenceTileDataset(
        root=data_cfg["root"],
        split=data_cfg.get("ref_bank_split", "ref_bank.txt"),
        split_dir=data_cfg["split_dir"],
        seed=seed,
        val_ratio=data_cfg.get("val_ratio", 0.2),
        ignore_index=data_cfg.get("ignore_index", 255),
        max_samples=data_cfg.get("max_ref_samples", -1),
    )

    train_pair_ds = ReferenceQueryPairDataset(train_ds, ref_ds, seed=seed)
    val_pair_ds = ReferenceQueryPairDataset(val_ds, ref_ds, seed=seed + 123)
    train_loader = DataLoader(
        train_pair_ds,
        batch_size=int(cfg["train"].get("batch_size", 4)),
        shuffle=True,
        num_workers=int(cfg["train"].get("num_workers", 0)),
        collate_fn=collate_pair_batch,
    )
    val_loader = DataLoader(
        val_pair_ds,
        batch_size=int(cfg["train"].get("batch_size", 4)),
        shuffle=False,
        num_workers=int(cfg["train"].get("num_workers", 0)),
        collate_fn=collate_pair_batch,
    )

    device = cfg["train"].get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model = ReferencePromptProposalNet(**cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"].get("lr", 1e-3)), weight_decay=float(cfg["train"].get("weight_decay", 1e-4)))

    work_dir = Path(cfg["train"]["work_dir"]).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    best_iou = -1.0
    history = []
    epochs = int(cfg["train"].get("epochs", 8))
    ignore_index = int(data_cfg.get("ignore_index", 255))

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        valid_train_batches = 0
        skipped_batches = 0
        for batch_idx, batch in enumerate(train_loader, start=1):
            query_inputs = batch["query"]["features"].to(device)
            ref_inputs = batch["reference"]["features"].to(device)
            ref_gt = batch["reference"]["gt"].to(device)
            query_gt = batch["query"]["gt"].to(device)
            ref_valid = batch["reference"]["valid_mask"].to(device)
            query_ids = batch["query"]["sample_id"]
            ref_ids = batch["reference"]["sample_id"]
            if not torch.isfinite(query_inputs).all() or not torch.isfinite(ref_inputs).all():
                skipped_batches += 1
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "batch": batch_idx,
                            "warning": "skip_non_finite_input",
                            "query_ids": query_ids,
                            "ref_ids": ref_ids,
                        },
                        ensure_ascii=False,
                    )
                )
                continue
            optimizer.zero_grad()
            output = model(query_inputs, ref_inputs, ref_gt, valid_mask=ref_valid)
            if not all(torch.isfinite(tensor).all() for tensor in [output.segmentation_logit, output.positive_logit, output.negative_logit, output.boundary_logit]):
                skipped_batches += 1
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "batch": batch_idx,
                            "warning": "skip_non_finite_output",
                            "query_ids": query_ids,
                            "ref_ids": ref_ids,
                        },
                        ensure_ascii=False,
                    )
                )
                continue
            losses = proposal_loss(output, query_gt, ignore_index=ignore_index)
            if not torch.isfinite(losses["loss"]):
                skipped_batches += 1
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "batch": batch_idx,
                            "warning": "skip_non_finite_loss",
                            "query_ids": query_ids,
                            "ref_ids": ref_ids,
                        },
                        ensure_ascii=False,
                    )
                )
                continue
            losses["loss"].backward()
            optimizer.step()
            train_loss += float(losses["loss"].item())
            valid_train_batches += 1

        model.eval()
        val_ious: List[float] = []
        with torch.no_grad():
            for batch in val_loader:
                query_inputs = batch["query"]["features"].to(device)
                ref_inputs = batch["reference"]["features"].to(device)
                ref_gt = batch["reference"]["gt"].to(device)
                query_gt = batch["query"]["gt"]
                ref_valid = batch["reference"]["valid_mask"].to(device)
                output = model(query_inputs, ref_inputs, ref_gt, valid_mask=ref_valid)
                pred = (torch.sigmoid(output.segmentation_logit) >= 0.5).long().cpu().squeeze(1)
                for idx in range(pred.shape[0]):
                    val_ious.append(binary_iou(pred[idx], query_gt[idx], ignore_index=ignore_index))
        mean_val_iou = float(np.mean(val_ious)) if val_ious else 0.0
        train_loss /= max(valid_train_batches, 1)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss if valid_train_batches > 0 else 0.0,
                "val_iou": mean_val_iou,
                "num_batches": len(train_loader),
                "valid_train_batches": valid_train_batches,
                "skipped_batches": skipped_batches,
            }
        )
        print(json.dumps(history[-1], ensure_ascii=False))
        torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch}, work_dir / "last_reference_prompt.pth")
        if mean_val_iou > best_iou:
            best_iou = mean_val_iou
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch, "best_val_iou": best_iou}, work_dir / "best_reference_prompt.pth")

    (work_dir / "reference_prompt_history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
