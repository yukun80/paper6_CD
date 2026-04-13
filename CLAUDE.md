# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SAR (Synthetic Aperture Radar) flood change detection research project. The active algorithm is **ChangeDINO** adapted for SAR waterlogging/flood detection.

**Active codebase:** `ChangeDINO-main/` only. Everything in `baselines/`, `GeoSA-BaSA-main/`, `sam_road-main/`, `panopticon/`, `exp_template/`, `AdaptOVCD-main/`, `dinov3_RS_CD/` is historical — not the current workflow.

## Commands

Run from repository root unless noted.

**Dataset preparation:**
```bash
python ChangeDINO-main/scripts/prepare_fused_sar_cd_dataset.py \
  --s1gfloods-root datasets/S1GFloods \
  --varfloods-root datasets/VarFloods \
  --out-root datasets/S1GFloods_CD_DINO \
  --tile-size 256 --stride 128 --train-ratio 0.9 --seed 42 --overwrite

python ChangeDINO-main/scripts/compute_s1gfloods_cd_stats.py \
  --data-root datasets/S1GFloods_CD_DINO --split train \
  --output datasets/S1GFloods_CD_DINO/channel_stats_s1gfloods_train.json
```

**Training (SAR mainline):**
```bash
cd ChangeDINO-main
BACKBONE_WEIGHT=pretrained/efficientnet_b0_ra-3dd342df.pth bash trainval_s1gfloods.sh
```

**Training (optical datasets):**
```bash
cd ChangeDINO-main
python trainval.py --name LEVIR-ChangeDINO --gpu_ids 0 --dataset LEVIR-CD --batch_size 16 --fpn_channels 128 --num_epochs 100 --lr 5e-4
```

**Testing/evaluation:**
```bash
cd ChangeDINO-main
python test.py --name <resolved_run_name> --dataset S1GFloods_CD_DINO \
  --dataroot ../datasets --dataset_mode sar \
  --stats_file ../datasets/S1GFloods_CD_DINO/channel_stats_s1gfloods_train.json \
  --gpu_ids 0 --save_test
```

**Whole-scene tiled inference (GF3 Henan):**
```bash
python ChangeDINO-main/scripts/prepare_gf3_henan_infer.py \
  --src-root datasets/GF3_Henan \
  --pre-image Pre_Zhengzhou_ascending_s1_radmatch.tif \
  --post-image Post_Zhengzhou_descending_clip.tif \
  --out-root datasets/GF3_Henan_CD_infer \
  --tile-size 256 --stride 128 --overwrite

python ChangeDINO-main/scripts/infer_gf3_henan_tiles.py \
  --tiles-root datasets/GF3_Henan_CD_infer \
  --checkpoint ChangeDINO-main/checkpoints/<resolved_run_name>/<resolved_run_name>_efficientnet_b0_best.pth \
  --stats_file datasets/S1GFloods_CD_DINO/channel_stats_s1gfloods_train.json \
  --gpu_ids 0 --batch_size 8 \
  --output-dir ChangeDINO-main/outputs/gf3_henan
```

**Syntax check before training:**
```bash
python -m py_compile ChangeDINO-main/model/ChangeDINO.py
```

**Optional dependency:**
```bash
pip install kornia
```

## Model Architecture

**ChangeDINO** is a dual-branch change detection network:

- **CNN Backbone:** EfficientNet-B0 (default) or MobileNetV2 — produces native p1–p5 feature pyramid. `convnextv2_nano` is deprecated.
- **Vision Foundation Model:** DINOv3 (ViT-S16/B16/L16). Weights at `ChangeDINO-main/dinov3/weights/`.
- **Collaboration mode:** `multilevel_v2` — DinoTokenBridge on p3/p2 + P1DinoSemanticGate.
- **Alignment:** Deformable Soft-Alignment on p1/p2/p3 levels.
- **Difference module:** ContrastAwareDiff with spatial+channel gate.
- **Gating:** DynamicMicroGate (3-route tiny prior: mean_abs_p1, mean_abs_p2_up, signed local contrast delta).
- **Refiner:** HybridRefiner with FloodTopoRouter for topology-aware flood boundary refinement.

Pretrained weight paths:
- EfficientNet-B0: `ChangeDINO-main/pretrained/efficientnet_b0_ra-3dd342df.pth`
- DINOv3 ViT-S16: `ChangeDINO-main/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth`

Checkpoint naming convention: `checkpoints/<name>-YYYYMMDD/<name>-YYYYMMDD_efficientnet_b0_best.pth`

## Key Configuration

All CLI flags live in `ChangeDINO-main/option.py`. Notable flags:
- `--dataset_mode sar` — disables saturation jitter, uses milder augmentation, requires `--stats_file`
- `--backbone` / `--backbone_weight` — CNN backbone selection
- `--dino_arch` / `--dino_weight` — DINOv3 variant
- `--topo_*` flags — topology branch parameters (grid size, hidden dim, neighbor mode, long offsets, n-hops, loss weight)
- `--branch_consistency_weight`, `--consistency_warmup_epochs` — consistency regularization
- `--best_metric`, `--eval_fg_threshold` — evaluation and model selection

Keep `--dataset` and `--stats_file` naming consistent with the actual output directory.

## Dataset Structure

**S1GFloods raw:** `datasets/S1GFloods/{A/, B/, Label/}`

**S1GFloods prepared (ChangeDINO-ready):** `datasets/S1GFloods_CD_DINO/{train,val}/{A,B,label}/` — no `test/` split.

**Optical CD datasets (WHU-CD, LEVIR-CD, etc.):** `datasets/<name>/{train,val,test}/{A/,B/,label/}`

**GF3 inference tiles:** `datasets/GF3_Henan_CD_infer/test/{A,B}/` (PNG, percentile-stretched 2/98) + `test_tif/` (float32 GeoTIFF with georeferencing)

## Core Files to Check Before Editing

- `ChangeDINO-main/option.py`
- `ChangeDINO-main/model/ChangeDINO.py`
- `ChangeDINO-main/model/create_ChangeDINO.py`
- `ChangeDINO-main/data/cd_dataset.py`
- `ChangeDINO-main/data/transform.py`
- `ChangeDINO-main/trainval.py`
- `ChangeDINO-main/trainval_s1gfloods.sh`
- `ChangeDINO-main/scripts/prepare_sar_scene_infer.py`
- `ChangeDINO-main/scripts/infer_sar_scene_tiles.py`

## Testing Guidelines

- For dataset script changes: run `--dry-run` first, then a short real execution on a small sample.
- For ChangeDINO code changes: run `python -m py_compile` on modified files, then a minimal train/eval sanity check before long jobs.

## Coding Style

- Python: 4-space indentation, PEP8, type hints for new utilities.
- Add concise Chinese comments/docstrings for key classes, key methods, and non-trivial code blocks.
- Prefer runtime arguments (`--data-root`, `--work-dir`) over hard-coded absolute paths.

## Commit Guidelines

Format: `scope: change summary` (e.g., `changedino: refine s1gfloods dataloader checks`). Chinese or English both accepted.

Do not commit datasets, model checkpoints, or large generated artifacts.

## Deprecated — Do Not Use

- `convnextv2_nano` backbone checkpoints
- UrbanSARFloods 12-channel SAR order conventions
- Hierarchical `floodness/flood_type` label workflow
- `pos_mIoU`-first model-selection policy
- Open-CD / Panopticon / exp_template as default training workflow
- `HSBA-flood/` directory (removed)
- `prepare_s1_henan_infer.py` / `infer_s1_henan_tiles.py` (replaced by generic SAR chain)
