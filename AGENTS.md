# Repository Guidelines

## Project Structure & Module Organization
This repository currently uses **ChangeDINO-based change detection** as the active algorithm line.

- `ChangeDINO-main/`: current primary training/inference codebase.
- `ChangeDINO_raw/`: clean backup copy of `ChangeDINO-main` algorithm code.
- `datasets/`: data assets and preprocessing scripts.
  - `datasets/S1GFloods/`: raw S1GFloods source data.
  - `datasets/S1GFloods_CD_DINO/`: prepared ChangeDINO-ready split data.
  - `datasets/S1_Henan/`, `datasets/S1_Henan_CD_infer/`: Henan scene data and tiled inference inputs.
  - `datasets/GF3_Henan/`: GF3 data asset only (not bound to deprecated prompt/SAM/PPO pipeline).
  - `datasets/script/`: dataset conversion/preprocessing utilities.
- `doc/`: experiment logs and archived notes.
- `baselines/`, `panopticon/`, `exp_template/`, `AdaptOVCD-main/`, `dinov3_RS_CD/`: kept in-tree as historical/auxiliary code, **not current default workflow**.

## Build, Test, and Development Commands
Run commands from repository root unless noted.

```bash
# Prepare S1GFloods flat data to ChangeDINO CD layout
python ChangeDINO-main/scripts/prepare_s1gfloods_cd.py \
  --src-root datasets/S1GFloods \
  --out-root datasets/S1GFloods_CD_DINO \
  --seed 42 \
  --overwrite

# Compute train split channel stats
python ChangeDINO-main/scripts/compute_s1gfloods_cd_stats.py \
  --data-root datasets/S1GFloods_CD_DINO \
  --split train \
  --output datasets/S1GFloods_CD_DINO/channel_stats_s1gfloods_train.json

# Train / validate (ChangeDINO on S1GFloods)
bash ChangeDINO-main/trainval_s1gfloods.sh

# Whole-scene S1 Henan tiled inference
python ChangeDINO-main/scripts/prepare_s1_henan_infer.py \
  --src-root datasets/S1_Henan \
  --pre-image Zhengzhou_S1GRD_ASCENDING_VH_pre.tif \
  --post-image Zhengzhou_S1GRD_ASCENDING_VH_Post.tif \
  --out-root datasets/S1_Henan_CD_infer \
  --tile-size 256 \
  --stride 128 \
  --overwrite

python ChangeDINO-main/scripts/infer_s1_henan_tiles.py \
  --tiles-root datasets/S1_Henan_CD_infer \
  --checkpoint ChangeDINO-main/checkpoints/S1GFloods-ChangeDINO/S1GFloods-ChangeDINO_mobilenetv2_best.pth \
  --stats_file datasets/S1GFloods_CD_DINO/channel_stats_s1gfloods_train.json \
  --gpu_ids 0 \
  --batch_size 8 \
  --output-dir ChangeDINO-main/outputs/s1_henan

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
  - `ChangeDINO-main/model/create_ChangeDINO.py`
  - `ChangeDINO-main/trainval.py`
  - `ChangeDINO-main/run.py`
  - `ChangeDINO-main/trainval_s1gfloods.sh`
- `S1GFloods` labels are binary flood-change labels; dataset preparation scripts convert mask semantics to training-ready format.
- Keep `--dataset` and `--stats_file` naming consistent with the actual output directory.

## Deprecated Context (Do Not Use)
- The following are deprecated and must not be used as current spec:
  - UrbanSARFloods 12-channel SAR order conventions.
  - Hierarchical `floodness/flood_type` label workflow.
  - `pos_mIoU`-first model-selection policy from old pipelines.
  - Open-CD / Panopticon / exp_template as default training workflow.
  - PPO/prompt/SAM-related historical attempts.

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
