# Repository Guidelines

## Project Structure & Module Organization
This repository currently uses **ChangeDINO-based change detection** as the active algorithm line.

- `ChangeDINO-main/`: current primary training/inference codebase.
- `baselines/ChangeDINO_raw/`: backup snapshot of an older ChangeDINO implementation, kept only for reference.
- `datasets/`: data assets and preprocessing scripts.
  - `datasets/S1GFloods/`: raw S1GFloods source data.
  - `datasets/S1GFloods_CD_DINO/`: prepared ChangeDINO-ready split data.
  - `datasets/GF3_Henan/`, `datasets/GF3_Henan_CD_infer/`: current GF3 Henan source data and tiled inference inputs.
  - `datasets/GF3_Zhuozhou/`, `datasets/GF3_Zhuozhou_CD_infer/`: GF3 Zhuozhou source data and tiled inference inputs.
  - `datasets/VarFloods/`: auxiliary SAR data source used by the fused training-set builder.
  - `datasets/script/`: dataset conversion/preprocessing utilities.
- `doc/`: experiment logs and archived notes.
- `baselines/`, `panopticon/`, `exp_template/`, `AdaptOVCD-main/`, `dinov3_RS_CD/`, `sam_road-main/`: kept in-tree as historical/auxiliary code, **not current default workflow**.
- `HSBA-flood/`: removed from the repository and must not be referenced as a current path or workflow component.

## Build, Test, and Development Commands
Run commands from repository root unless noted.

```bash
# Prepare fused S1GFloods + VarFloods SAR CD dataset
python ChangeDINO-main/scripts/prepare_fused_sar_cd_dataset.py \
  --s1gfloods-root datasets/S1GFloods \
  --varfloods-root datasets/VarFloods \
  --out-root datasets/S1GFloods_CD_DINO \
  --tile-size 256 \
  --stride 128 \
  --train-ratio 0.9 \
  --seed 42 \
  --overwrite

# Compute train split channel stats
python ChangeDINO-main/scripts/compute_s1gfloods_cd_stats.py \
  --data-root datasets/S1GFloods_CD_DINO \
  --split train \
  --output datasets/S1GFloods_CD_DINO/channel_stats_s1gfloods_train.json

# Train / validate (ChangeDINO on S1GFloods)
bash ChangeDINO-main/trainval_s1gfloods.sh

# Whole-scene GF3 Henan tiled inference
python ChangeDINO-main/scripts/prepare_gf3_henan_infer.py \
  --src-root datasets/GF3_Henan \
  --pre-image Pre_Zhengzhou_ascending_s1_radmatch.tif \
  --post-image Post_Zhengzhou_descending_clip.tif \
  --out-root datasets/GF3_Henan_CD_infer \
  --tile-size 256 \
  --stride 128 \
  --overwrite

python ChangeDINO-main/scripts/infer_gf3_henan_tiles.py \
  --tiles-root datasets/GF3_Henan_CD_infer \
  --checkpoint ChangeDINO-main/checkpoints/S1GFloods-ChangeDINO-vitl16/S1GFloods-ChangeDINO-vitl16_convnextv2_nano_best.pth \
  --stats_file datasets/S1GFloods_CD_DINO/channel_stats_s1gfloods_train.json \
  --gpu_ids 0 \
  --batch_size 8 \
  --output-dir ChangeDINO-main/outputs/gf3_henan

# Optional: force fixed-value stretch for pre-image only
python ChangeDINO-main/scripts/prepare_gf3_henan_infer.py \
  --src-root datasets/GF3_Henan \
  --pre-image Pre_Zhengzhou_ascending_s1_radmatch.tif \
  --post-image Post_Zhengzhou_descending_clip.tif \
  --out-root datasets/GF3_Henan_CD_infer_preclip \
  --tile-size 256 \
  --stride 128 \
  --pre-stretch-mode value \
  --pre-value-min 0.2 \
  --pre-value-max 2.0 \
  --overwrite

# Default ConvNeXtV2 nano local weight
# Place at: ChangeDINO-main/pretrained/convnextv2_nano_22k_224_ema.pt

# Optional dependency
pip install kornia
```

## Coding Style & Naming Conventions
- Python: 4-space indentation, PEP8-compatible style, and type hints for new utilities.
- Keep script naming descriptive and task-specific.
- For new or modified code, add concise Chinese comments/docstrings for key classes, key methods, and non-trivial code blocks.
- Keep comments focused on intent and logic; avoid redundant comments.

## Testing Guidelines
- No single root test suite is enforced; validate within touched module.
- For dataset script changes:
  - run `--dry-run` first;
  - then run a short real execution on a small sample.
- For ChangeDINO code changes:
  - run `python -m py_compile` on modified Python files;
  - run one minimal train/eval sanity check before long jobs.

## ChangeDINO Notes
- Core files to check before editing:
  - `ChangeDINO-main/option.py`
  - `ChangeDINO-main/data/cd_dataset.py`
  - `ChangeDINO-main/data/transform.py`
  - `ChangeDINO-main/scripts/prepare_sar_scene_infer.py`
  - `ChangeDINO-main/scripts/infer_sar_scene_tiles.py`
  - `ChangeDINO-main/model/ChangeDINO.py`
  - `ChangeDINO-main/model/create_ChangeDINO.py`
  - `ChangeDINO-main/trainval.py`
  - `ChangeDINO-main/run.py`
  - `ChangeDINO-main/trainval_s1gfloods.sh`
- `S1GFloods` labels are binary flood-change labels; dataset preparation scripts convert mask semantics to training-ready format.
- Keep `--dataset` and `--stats_file` naming consistent with the actual output directory.
- Current default CNN backbone is `convnextv2_nano`.
- Default local backbone weight path is `ChangeDINO-main/pretrained/convnextv2_nano_22k_224_ema.pt`.
- Only `convnextv2_nano` and `mobilenetv2` are supported; `resnet18d` must not be used as current spec.
- Current SAR refiner is `FloodTopoRouter`; topology-related options live in `option.py` (`--topo_grid_size`, `--topo_hidden_dim`, `--topo_neighbor_k`, `--topo_n_hops`, `--topo_loss_weight`).
- Current whole-scene inference entrypoint is the generic SAR tiling/stitching chain (`prepare_*_infer.py` -> `infer_gf3_henan_tiles.py` / `infer_sar_scene_tiles.py`), not the removed `prepare_s1_henan_infer.py` / `infer_s1_henan_tiles.py`.
- PNG tile export now supports both global and pre/post-specific stretch controls; by default both branches still use percentile stretch `2/98`.

## Deprecated Context (Do Not Use)
- The following are deprecated and must not be used as current spec:
  - UrbanSARFloods 12-channel SAR order conventions.
  - Hierarchical `floodness/flood_type` label workflow.
  - `pos_mIoU`-first model-selection policy from old pipelines.
  - Open-CD / Panopticon / exp_template as default training workflow.
  - PPO/prompt/SAM-related historical attempts.
  - Any removed directories or workflows such as `HSBA-flood/`.

## Commit & Pull Request Guidelines
- Keep commit messages concise and specific (Chinese or English).
- Recommended format: `scope: change summary` (example: `changedino: refine s1gfloods dataloader checks`).
- PRs should include:
  - purpose and impacted paths,
  - exact reproduction commands,
  - key logs/metrics locations,
  - dataset/environment assumptions.
- Do not commit datasets, model checkpoints, or large generated artifacts.

## Security & Configuration Tips
- Prefer runtime arguments (`--data-root`, `--work-dir`) over hard-coded absolute paths.
- Keep secrets/tokens out of tracked files and shell scripts.
