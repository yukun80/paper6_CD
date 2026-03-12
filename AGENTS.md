# Repository Guidelines

## Project Structure & Module Organization
This repository is a multi-baseline workspace for remote sensing change detection.

- `baselines/open-cd/`: primary training framework used in current workflows (FC-Siam, custom UrbanSARFloods configs/scripts).
- `datasets/`: data assets and tooling.
  - `datasets/urban_sar_floods/`: raw source data.
  - `datasets/urban_sar_floods_ch12_256/`: 256 切片、12 通道输入数据（含层次化标签可选目录）。
  - `datasets/urban_sar_floods_CD/`: converted CD/SCD-ready dataset.
  - `datasets/GF3_Henan/`: GF3 河南灾前/灾后 HV 极化幅度图（当前仅保留为数据资产，不绑定已废弃的 prompt/SAM/PPO 实验链路）。
  - `datasets/script/`: conversion and preprocessing scripts.
- `doc/`: experiment logs and design notes (for example `doc/工作日志_2026-03-03.md`).
- `dinov3_RS_CD/`, `panopticon/`, `AdaptOVCD-main/`, `baselines/CMCDNet/`: auxiliary research codebases kept in-tree.

## Build, Test, and Development Commands
Run commands from repository root unless noted.

```bash
# Tile 512x512 UrbanSARFloods to 256x256 and expand SAR from 8ch to 12ch
python datasets/script/tile_urban_sar_floods_to_256_ch12.py --src-root datasets/urban_sar_floods --out-root datasets/urban_sar_floods_ch12_256 --overwrite --strict

# Rebuild hierarchical labels from GT(0/1/2):
# - floodness: BG->0, FO/FU->1
# - flood_type: FO->0, FU->1, BG->255(ignore)
python datasets/script/rebuild_urban_sar_floods_hier_labels.py --data-root datasets/urban_sar_floods_ch12_256 --ignore-index 255 --overwrite --strict

# Convert raw UrbanSARFloods to CD layout
python datasets/script/convert_urban_sar_floods_to_cd.py --src-root datasets/urban_sar_floods --out-root datasets/urban_sar_floods_CD --overwrite --strict

# Check OpenMMLab environment compatibility
./baselines/open-cd/scripts/urban_sar_floods/run_fcsiam_conc_3c.sh check-env

# Smoke training / full training / evaluation
./baselines/open-cd/scripts/urban_sar_floods/run_fcsiam_conc_3c.sh smoke-train
./baselines/open-cd/scripts/urban_sar_floods/run_fcsiam_conc_3c.sh full-train
./baselines/open-cd/scripts/urban_sar_floods/run_fcsiam_conc_3c.sh eval

# Panopticon UrbanSARFloods training / eval
PYTHONPATH=panopticon python panopticon/urban_floods/train.py --config-file panopticon/configs/urban_floods_seg.yaml
PYTHONPATH=panopticon python panopticon/urban_floods/eval.py --config-file panopticon/configs/urban_floods_seg.yaml --ckpt <ckpt_path> --split val

# Panopticon Hierarchical PanFlood-Adapter training / eval
PYTHONPATH=panopticon python panopticon/urban_floods_hier/train_hier.py --config-file panopticon/configs/urban_floods_hier_seg.yaml
PYTHONPATH=panopticon python panopticon/urban_floods_hier/eval_hier.py --config-file panopticon/configs/urban_floods_hier_seg.yaml --ckpt <ckpt_path> --split val

# Compute robust 12ch normalization stats (train split only)
python datasets/script/compute_urban_sar_floods_ch12_stats.py --data-root datasets/urban_sar_floods_ch12_256 --split-file Train_dataset.txt --strict

# Check samples that become all-ignore after center crop
python datasets/script/check_urban_sar_floods_ignore_samples.py --data-root datasets/urban_sar_floods_256 --crop-size 252 --num-workers 8

# exp_template 单模型训练 / 评估 / 推理可视化（以 unet 为例）
python exp_template/train.py --config-file exp_template/config/unet.yaml
python exp_template/eval.py --config-file exp_template/config/unet.yaml --ckpt exp_template/work_dir/<run_name>/checkpoints/best.pth --split val
python exp_template/visualize.py --config-file exp_template/config/unet.yaml --ckpt exp_template/work_dir/<run_name>/checkpoints/best.pth --split val --max-samples 50

# exp_template 一次性批训练全部对比模型（失败继续，最后汇总）
bash exp_template/train_all_models.sh

# TensorBoard（可选，若提示未安装）
pip install tensorboard
```

## Coding Style & Naming Conventions
- Python: 4-space indentation, PEP8-compatible style, and type hints for new utilities.
- Keep config/script naming descriptive and task-specific, e.g. `fc_siam_conc_*_urban_sar_floods_3c.py`, `run_fcsiam_conc_3c.sh`.
- For new or modified code, add concise Chinese comments/docstrings for key classes, key methods, and non-trivial code blocks to improve readability and handoff efficiency.
- Keep comments focused on intent and logic; avoid redundant comments that only restate obvious code behavior.
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

## Panopticon UrbanSARFloods Notes
- Downstream task entry:
  - `panopticon/urban_floods/train.py`
  - `panopticon/urban_floods/eval.py`
  - `panopticon/configs/urban_floods_seg.yaml`
- Hierarchical downstream task entry:
  - `panopticon/urban_floods_hier/train_hier.py`
  - `panopticon/urban_floods_hier/eval_hier.py`
  - `panopticon/configs/urban_floods_hier_seg.yaml`
- Dataset check utility:
  - `datasets/script/check_urban_sar_floods_ignore_samples.py`
  - `ignore_check_train.txt` / `ignore_check_val.txt` now output `SAR` paths.
  - `ignore_check_report.json` keeps both `sar_path` and `gt_path`.
- `Train_dataset.txt` / `Valid_dataset.txt` are GT-path lists by design; if deleting bad samples from split, remove GT entries.
- 层次化标签重构脚本：
  - `datasets/script/rebuild_urban_sar_floods_hier_labels.py`
  - 输出并行目录：`GT_floodness/` 与 `GT_flood_type/`（不覆盖原 `GT/`）
  - `GT_flood_type` 默认忽略值：`255`（与现有 `ignore_index` 配置一致）
- 12 通道顺序（`urban_sar_floods_ch12_256`，VH/VV 版）：
  - `pre_coh_vh, pre_coh_vv, co_coh_vh, co_coh_vv, pre_int_vh, pre_int_vv, co_int_vh, co_int_vv, dcoh_vh, dcoh_vv, dint_vh, dint_vv`
  - 层次化配置需与该顺序对齐：`channel_ids/time_ids/feature_type_ids/temporal_role_ids/polarization_ids`
- 12 通道统计量脚本：
  - `datasets/script/compute_urban_sar_floods_ch12_stats.py`
  - 默认使用 `Train_dataset.txt`，输出 `datasets/urban_sar_floods_ch12_256/channel_stats_ch12_train.json`
  - 优先使用上述 JSON 的 `recommended_config_fields.mean/std` 回填配置
- Hierarchical 评估与监督口径（`urban_floods_hier`）：
  - 评估主指标：`pos_mIoU`（正类 `[1, 2]`），并记录 `IoU_1/IoU_2/mIoU/pos_mF1`
  - 三标签使用：`main_label` 用于三分类一致性 CE；`floodness_label` 用于洪水二分类；`flood_type_label` 仅在非 `ignore_index` 像素参与类型损失
  - 最终三类概率重组：`P(BG)=1-Pf`, `P(FO)=Pf*(1-Pu)`, `P(FU)=Pf*Pu`，预测为 `argmax`
- Best-checkpoint selection for panopticon urban floods is configured in YAML:
  - default `selection_metric: pos_mIoU` (positive classes `[1, 2]`)
  - tie-breakers: `IoU_2`, then `mIoU`.
- xformers 运行提示：
  - `TypedStorage is deprecated` 多为 xformers 内部兼容告警，通常不导致训练失败
  - 是否真正报错以 traceback 和进程退出码为准

## exp_template UrbanSARFloods Notes
- 入口脚本：
  - `exp_template/train.py`
  - `exp_template/eval.py`
  - `exp_template/visualize.py`
  - `exp_template/train_all_models.sh`
- 模型配置：
  - `exp_template/config/{resnet_fcn,unet,deeplabv3plus,pspnet,swin_uperlite}.yaml`
- 数据与标签口径：
  - 数据目录：`datasets/urban_sar_floods_ch12_256`
  - 标签直接使用原始三分类 `0/1/2`，不使用二分类构造标签
- 评估指标口径：
  - 包含 `IoU / Precision / Recall / F1 / OA`
  - 同时记录 `mIoU/mPrecision/mRecall/mF1` 与 `pos_mIoU/pos_mPrecision/pos_mRecall/pos_mF1`
- TensorBoard 约定：
  - 默认开启：`train.tensorboard.enabled: true`
  - 默认布局：`train.tensorboard.layout_style: grouped`（按 Loss 和 Validation Metrics 分组，避免单列堆叠）
  - 若环境缺失 TensorBoard，请先安装 `tensorboard`

## Archived Attempts
- `PPO-main/` 与 `sar_prompt_flood/` 相关实验链路已从当前仓库移除，因为无法继续使用。
- 相关命令、checkpoint、配置、日志结论与设计假设均已废弃；后续分析、实现与协作时不要再把这两次尝试纳入当前上下文。
- `datasets/GF3_Henan/` 目前仅作为数据资产保留；若未来重启该方向，应基于新的独立方案重新立项与记文档。

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
