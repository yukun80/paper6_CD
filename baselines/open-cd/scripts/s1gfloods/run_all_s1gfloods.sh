#!/usr/bin/env bash
# 仓库根目录、opencd 环境；每一步成功后再执行下一步：
# CUDA_VISIBLE_DEVICES=0 bash baselines/open-cd/scripts/s1gfloods/run_all_s1gfloods.sh check-env
# CUDA_VISIBLE_DEVICES=0 bash baselines/open-cd/scripts/s1gfloods/run_all_s1gfloods.sh smoke-train
# CUDA_VISIBLE_DEVICES=0 bash baselines/open-cd/scripts/s1gfloods/run_all_s1gfloods.sh full-train
# 默认9模型（排除STANet、ChangeStar），主权重best_FloodIoU；--save-best mIoU可切换主指标。
# 默认 --batch-scale 2：普通模型16、TTP 4，20000迭代；--batch-scale 1恢复8/2和40000迭代。
# 新 batch 配置待手动验收，旧短训练验收不适用于新配置。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python}" "${SCRIPT_DIR}/batch_train.py" "$@"
