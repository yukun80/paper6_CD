#!/usr/bin/env bash
# 在仓库根目录，conda activate hacqi 后依次执行：
# CUDA_VISIBLE_DEVICES=0 bash baselines/ChangeDINO_raw/trainval_s1gfloods.sh check-env
# CUDA_VISIBLE_DEVICES=0 bash baselines/ChangeDINO_raw/trainval_s1gfloods.sh smoke-train
# CUDA_VISIBLE_DEVICES=0 bash baselines/ChangeDINO_raw/trainval_s1gfloods.sh full-train
# 可覆盖 DATASET_NAME、DATA_ROOT（仓库相对路径）、RUN_NAME、BATCH_SIZE、NUM_EPOCHS、LR、SEED。
# 不传模式只执行 check-env；正式训练须显式 full-train。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -eq 0 ]]; then set -- check-env; fi
exec "${PYTHON:-python}" "${SCRIPT_DIR}/scripts/prepare_training.py" "$@"
