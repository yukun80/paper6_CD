# Repository Guidelines

## Project Structure & Module Organization
This repository is a multi-baseline workspace for remote sensing change detection.

- `baselines/open-cd/`: primary training framework used in current workflows (FC-Siam, custom UrbanSARFloods configs/scripts).
- `datasets/`: data assets and tooling.
  - `datasets/urban_sar_floods/`: raw source data.
  - `datasets/urban_sar_floods_CD/`: converted CD/SCD-ready dataset.
  - `datasets/script/`: conversion and preprocessing scripts.
- `doc/`: experiment logs and design notes (for example `doc/工作日志_2026-03-03.md`).
- `dinov3_RS_CD/`, `panopticon/`, `AdaptOVCD-main/`, `baselines/CMCDNet/`: auxiliary research codebases kept in-tree.

## Build, Test, and Development Commands
Run commands from repository root unless noted.

```bash
# Convert raw UrbanSARFloods to CD layout
python datasets/script/convert_urban_sar_floods_to_cd.py --src-root datasets/urban_sar_floods --out-root datasets/urban_sar_floods_CD --overwrite --strict

# Check OpenMMLab environment compatibility
./baselines/open-cd/scripts/urban_sar_floods/run_fcsiam_conc_3c.sh check-env

# Smoke training / full training / evaluation
./baselines/open-cd/scripts/urban_sar_floods/run_fcsiam_conc_3c.sh smoke-train
./baselines/open-cd/scripts/urban_sar_floods/run_fcsiam_conc_3c.sh full-train
./baselines/open-cd/scripts/urban_sar_floods/run_fcsiam_conc_3c.sh eval
```

## Coding Style & Naming Conventions
- Python: 4-space indentation, PEP8-compatible style, and type hints for new utilities.
- Keep config/script naming descriptive and task-specific, e.g. `fc_siam_conc_*_urban_sar_floods_3c.py`, `run_fcsiam_conc_3c.sh`.
- Style tools vary by submodule:
  - `dinov3_RS_CD`: `ruff` (line length 120).
  - `panopticon`: `black` (line length 120).
  - `baselines/CMCDNet`: `yapf`/`isort` settings in `setup.cfg`.

## Testing Guidelines
- No single root test suite is enforced; validate changes in the touched module.
- Open-CD changes: run `smoke-train` before long runs.
- Dataset pipeline changes: run converter with `--dry-run` first, then a short train/eval sanity run.
- CMCDNet-specific updates can use:
  - `pytest baselines/CMCDNet/tests -q`

## Open-CD UrbanSARFloods Notes
- Core customized files to check before editing:
  - `baselines/open-cd/configs/fcsn/fc_siam_diff_512x512_60k_urban_sar_floods_3c.py`
  - `baselines/open-cd/opencd/datasets/urban_sar_floods_cd.py`
  - `baselines/open-cd/opencd/datasets/transforms/loading.py`
  - `baselines/open-cd/scripts/urban_sar_floods/run_fcsiam_conc_3c.sh`
- Open-CD algorithm pruning (2026-03-03):
  - Removed from framework code/config: `FC-EF`, `FC-Siam-conc`, `STANet`, `SNUNet`, `ChangeFormer`, `TinyCD`, `TinyCDv2`, `HANet`, `CGNet`, `TTP`.
  - Keep this in mind when reusing old configs or importing removed model classes.
- Keep invalid-value masking enabled for `.npy` inputs (`MultiImgApplyInvalidMask`) to avoid NaN loss from source data.
- For checkpoint selection, use prefixed metrics (`val/mIoU`, `val/mFscore`) to match evaluator outputs.
- After any Open-CD workflow/config update, synchronize:
  - `doc/工作日志_2026-03-03.md` (what changed, why, validation result)
  - `AGENTS.md` (if contributor workflow/commands changed)

## Commit & Pull Request Guidelines
- Current history uses short one-line commit messages (Chinese or English). Keep them concise and specific.
- Recommended format: `scope: change summary` (example: `open-cd: add invalid-mask transform for NaN handling`).
- PRs should include:
  - purpose and impacted paths,
  - exact reproduction commands,
  - key metrics/log locations (for training changes),
  - any dataset or environment assumptions.
- Do not commit datasets, model checkpoints, or large generated artifacts.

## Security & Configuration Tips
- Prefer runtime arguments (`--data-root`, `--work-dir`) over hard-coded absolute paths.
- Keep secrets/tokens out of tracked files and shell scripts.
