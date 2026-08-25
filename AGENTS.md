# Repository Guidelines

## Project Role & Current Priority
This repository is a combined flood-change-detection, flood-depth-estimation, and writing workspace.

- `HA-CQI/` is the current primary algorithm code path for SAR flood change detection.
- `HA-CQI-CFDepth/` is the integrated HA-CQI + CFDepth package for paper-facing reproduction, scene prediction, and CFDepth implementation reference.
- `paper6_en/` is the English manuscript workspace.
- `patent/` is the Chinese invention patent application workspace.
- `technical proposal/` is the project technical-solution/proposal workspace.
- `baselines/`, `demo/`, and historical snapshots are for comparison, visualization, or archived reference only.

Keep these artifact boundaries explicit. Do not mix manuscript prose, patent claims, and technical-proposal language unless the user asks for a conversion between them.

## Project Structure & Module Organization
- `HA-CQI/`: active SAR flood-change detection implementation.
  - `model/architectures/ha_cqi.py`: HA-CQI model assembly.
  - `model/engine.py`: training/inference engine, losses, checkpoint metadata.
  - `model/modules/`: harmonized alignment, DINO adapter, CQI, semantic encoding, attention blocks.
  - `model/decode_heads/`: OSCD state-space dense change decoder and auxiliary heads.
  - `data/`: SAR change-detection dataset and transforms.
  - `scripts/`: whole-scene SAR tile inference, label preparation, and mosaic utilities.
  - `model_design/paper_narrative.md`: model-story reference for paper-aligned algorithm changes.
- `HA-CQI-CFDepth/`: integrated release-style path for HA-CQI and CFDepth.
  - `train.py`, `train.sh`: training entrypoints.
  - `predict.py`, `predict.sh`: tiled SAR scene prediction and mosaic reconstruction.
  - `CFDepth/CFDepth_GEE.txt`: authoritative CFDepth implementation reference for GEE-based depth estimation.
  - `datasets/train_set/`: packaged training split manifests and stats.
- `datasets/`: source data and prepared SAR change-detection data.
  - `S1GFloods/`: raw S1GFloods data.
  - `S1GFloods_CD_DINO_BG_75_25/`: active prepared binary SAR change-detection dataset.
  - `S1GFloods_CD_DINO/`: archived pre-correction dataset; select explicitly only for historical analysis.
  - `S1GFloods_CD_DINO_/`: experiment-specific variant; select explicitly.
  - `GF3_Henan/`, `GF3_Henan_CD_infer/`: Zhengzhou/GF3 source and tiled inference inputs.
  - `GF3_Zhuozhou/`, `GF3_Zhuozhou_CD_infer/`: Zhuozhou/GF3 source and tiled inference inputs.
  - `VarFloods/`: auxiliary SAR data source for fused training-set construction.
  - `script/`: dataset conversion and preprocessing utilities.
- `paper6_en/`: Elsevier manuscript, BibTeX, figures, and tables. Main manuscript file is `elsarticle-template-harv_2.tex`.
- `patent/`: patent drafts, converted reference documents, figures, and prior-art references.
- `technical proposal/`: project technical proposal drafts; quote the path because it contains a space.
- `baselines/`: comparison and historical code paths, including `ChangeDINO-main`, `ChangeDINO_raw`, and `open-cd`.
- `demo/`: Zhengzhou/Zhuozhou comparison visualizations and paper/demo artifacts.

## Current Algorithm Notes
- The active change-detection model is **HA-CQI**, not ChangeDINO.
- The active HA-CQI stack is:
  - shared pre/post CNN-DINO semantic encoder;
  - EfficientNet-B2-only CNN-FPN pyramid;
  - frozen DINOv3 semantic features from explicit fusion layers `[2,8,11]` with fixed LVD ImageNet normalization;
  - Harmonized Alignment on shallow features;
  - optional deformable soft alignment;
  - default five-level Change Query Interaction; the `local_structural` ablation replaces only P1/P2 with query-free locally normalized change projection;
  - MMSCoPE-inspired Omni-Scale State-Space Change Decoder (OSCD);
  - multi-scale auxiliary supervision for small waterlogging and large inundation regions.
- Preserve HA-CQI's own architecture story. Do not back-port ChangeDINO Risk-Aware, HybridRefiner, topo-router, or micro-gate modules unless the user explicitly requests that experiment.
- HA-CQI selects `best_primary` by validation Flood IoU while jointly calibrating threshold on the configured grid. `0.40` is only a validation diagnostic threshold.
- Active code accepts checkpoint v2 only. The 20260427 checkpoints are disk archives and are intentionally unsupported.
- CFDepth is the depth-estimation method. Change detection provides an upstream flood extent/support-region input; it is not the patent core unless the user changes the invention scope.

## Build, Train, and Inference Commands
Run commands from repository root unless a command starts with `cd`.

### HA-CQI Training
```bash
cd HA-CQI
bash trainval_s1gfloods.sh
```

Useful overrides:

```bash
cd HA-CQI
DATASET_NAME=S1GFloods_CD_DINO_BG_75_25 \
DATA_ROOT=../datasets \
RUN_NAME=S1GFloods-HA-CQI-B2-vits16 \
BATCH_SIZE=12 \
NUM_WORKERS=8 \
LR=1e-4 \
AMP_DTYPE=bf16 \
EVAL_FG_THRESHOLD=0.40 \
bash trainval_s1gfloods.sh
```

Disable soft alignment for ablation:

```bash
cd HA-CQI
SOFT_ALIGNMENT=0 RUN_NAME=S1GFloods-HA-CQI-noalign bash trainval_s1gfloods.sh
```

### HA-CQI Whole-Scene Inference
```bash
python HA-CQI/scripts/infer_gf3_henan_tiles.py \
  --tiles-root datasets/GF3_Henan_CD_infer \
  --checkpoint HA-CQI/checkpoints/<b2_run>/<b2_run>_efficientnet_b2_best_primary.pth \
  --gpu_ids 0 \
  --batch_size 8 \
  --output-dir HA-CQI/outputs/gf3_henan
```

For Zhuozhou, switch `--tiles-root` and `--output-dir`:

```bash
python HA-CQI/scripts/infer_gf3_henan_tiles.py \
  --tiles-root datasets/GF3_Zhuozhou_CD_infer \
  --checkpoint HA-CQI/checkpoints/<b2_run>/<b2_run>_efficientnet_b2_best_primary.pth \
  --gpu_ids 0 \
  --batch_size 8 \
  --output-dir HA-CQI/outputs/gf3_zhuozhou
```

### HA-CQI Pair Inference
```bash
cd HA-CQI
python run.py \
  --checkpoint checkpoints/<b2_run>/<b2_run>_efficientnet_b2_best_primary.pth \
  --img_A /path/to/pre_image.tif \
  --img_B /path/to/post_image.tif \
  --output outputs/run_pred.png \
  --gpu_ids 0
```

### HA-CQI-CFDepth Training and Prediction
```bash
cd HA-CQI-CFDepth
bash train.sh
```

```bash
cd HA-CQI-CFDepth
CHECKPOINT=checkpoints/HA-CQI-vits16/HA-CQI-vits16_efficientnet_b0_best.pth \
BATCH_SIZE=8 \
THRESHOLD=0.40 \
bash predict.sh
```

Single-scene prediction:

```bash
cd HA-CQI-CFDepth
python predict.py \
  --tiles-root datasets/test_set_Zhengzhou \
  --checkpoint checkpoints/HA-CQI-vits16/HA-CQI-vits16_efficientnet_b0_best.pth \
  --stats_file datasets/train_set/channel_stats_s1gfloods_train.json \
  --gpu_ids 0 \
  --batch_size 8 \
  --threshold 0.40 \
  --output-dir outputs/test_set_Zhengzhou
```

### CFDepth
- Use `HA-CQI-CFDepth/CFDepth/CFDepth_GEE.txt` as the implementation reference for CFDepth.
- CFDepth expects a 0/1 flood mask or flood support region and DEM-derived constraints.
- Do not claim a Python depth-estimation pipeline exists unless the corresponding implementation is present.

### Manuscript
```bash
cd paper6_en
latexmk -xelatex elsarticle-template-harv_2.tex
```

If `latexmk` is unavailable, use the local LaTeX toolchain available in the environment and verify that references, figures, and tables resolve.

## Data, Weights, and Generated Artifacts
- Do not commit datasets, model checkpoints, pretrained weights, large generated rasters, or bulky rendered outputs.
- HA-CQI default local weights:
  - `HA-CQI/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth`
  - `HA-CQI/pretrained/efficientnet_b2_ra-bcdf34b7.pth`
- HA-CQI-CFDepth default local weights:
  - `HA-CQI-CFDepth/pretrained/dinov3_vits16_pretrain_lvd1689m-08c60483.pth`
  - `HA-CQI-CFDepth/pretrained/efficientnet_b0_ra-3dd342df.pth`
- Dataset stats must match the selected dataset root. Keep `--dataset`, `--dataroot`, and `--stats_file` consistent.
- Scene-inference directories must contain `tile_manifest.csv`, A/B tile PNGs, and valid-mask files before running tiled prediction.

## Coding Style & Naming Conventions
- Python uses 4-space indentation, PEP8-compatible layout, and type hints for new utilities.
- Add concise Chinese comments/docstrings for key classes, key methods, and non-trivial logic blocks.
- Keep comments focused on intent and behavior; avoid restating obvious assignments.
- Prefer runtime arguments such as `--data-root`, `--tiles-root`, `--checkpoint`, `--stats_file`, and `--output-dir` over hard-coded absolute paths.
- Keep script names descriptive and task-specific.
- Use structured readers for CSV, JSON, raster, and LaTeX/BibTeX content where practical.
- Preserve the checkpoint v2 contract: require `meta.format_version == 2`, `network`, `meta.model_config`, and an explicit or selected inference threshold.

## Validation Guidelines
- For modified Python files, run `python -m py_compile` on the touched files.
- For HA-CQI core edits:
  ```bash
  cd HA-CQI
  python -m py_compile \
    option.py \
    model/architectures/ha_cqi.py \
    model/modules/*.py \
    model/decode_heads/*.py \
    model/engine.py \
    trainval.py \
    test.py \
    run.py \
    scripts/diagnose_cross_domain_features.py \
    scripts/infer_sar_scene_tiles.py \
    scripts/infer_gf3_henan_tiles.py
  bash -n trainval_s1gfloods.sh trainval.sh
  ```
- For HA-CQI-CFDepth core edits:
  ```bash
  cd HA-CQI-CFDepth
  python -m py_compile \
    option.py \
    train.py \
    predict.py \
    model/architectures/ha_cqi.py \
    model/modules/*.py \
    model/decode_heads/*.py \
    model/engine.py
  bash -n train.sh predict.sh
  ```
- For dataset script changes, run `--dry-run` first when supported, then a short real execution on a small sample.
- For tiled inference changes, verify that `infer_report.json`, stitched probability/binary outputs, valid-mask handling, and no-data values are correct.
- For paper edits, compile the manuscript and check figure/table/reference numbering.
- For patent or proposal edits, run a manual consistency check for artifact type, source hierarchy, terminology, and whether protected claims drift beyond the intended method.

## Writing Artifact Rules
### `paper6_en/`
- Write in academic manuscript style, grounded in the current figures, tables, code, and experiment outputs.
- Use `paper6_en/elsarticle-template-harv_2.tex`, `reference.bib`, `figure/`, and `tables/` as the authoritative paper artifacts.
- Do not import patent claim wording into the paper.

### `patent/`
- Target artifact is a **Chinese invention patent application text**, not a technical disclosure.
- The CFDepth depth-estimation method in `paper6_en/elsarticle-template-harv_2.tex` is the core method source.
- `HA-CQI-CFDepth/CFDepth/CFDepth_GEE.txt` is the implementation check for CFDepth details.
- Existing documents under `patent/` are format references.
- `patent/参考文献/` is prior-art/background only, not the invention source.
- Do not write repository paths, paper titles, code names, or experiment-only details into the formal patent body.
- Do not accidentally protect the HA-CQI change-detection model when the user asks for the CFDepth water-depth invention.

### `technical proposal/`
- Treat this as a project technical-solution document, not a manuscript or patent application.
- Emphasize system architecture, implementation routes, datasets, validation, deployment assumptions, and deliverables.
- Keep claims and performance statements tied to available evidence; do not invent project results.
- Quote the path in shell commands: `'technical proposal'`.

## Baselines and Deprecated Context
- `baselines/ChangeDINO-main/` and `baselines/ChangeDINO_raw/` are historical/comparison paths, not the active workflow.
- `baselines/open-cd/` is for baseline comparison and reruns only.
- Do not refer to a root-level `ChangeDINO-main/` path as current; the active root-level algorithm path is `HA-CQI/`.
- Deprecated as current spec:
  - old ChangeDINO detector/refiner defaults;
  - UrbanSARFloods 12-channel conventions;
  - hierarchical `floodness/flood_type` label workflows;
  - `pos_mIoU`-first model-selection policy from older pipelines;
  - PPO/prompt/SAM historical attempts;
  - removed paths such as `HSBA-flood/`.

## Commit & Pull Request Guidelines
- Keep commit messages concise and specific, Chinese or English.
- Recommended format: `scope: change summary`.
- PRs should include purpose, impacted paths, exact reproduction commands, key logs/metrics, dataset assumptions, and environment assumptions.
- Do not commit datasets, checkpoints, pretrained weights, secrets, or large generated artifacts.

## Security & Configuration
- Keep secrets, API keys, and tokens out of tracked files and scripts.
- Prefer environment variables or runtime arguments for local machine paths.
- Before destructive cleanup of generated outputs, confirm the target path and preserve user-created artifacts unless explicitly told otherwise.
