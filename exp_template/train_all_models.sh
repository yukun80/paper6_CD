#!/usr/bin/env bash
set -euo pipefail

# 一次性顺序训练 exp_template 下全部对比模型。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

# 可选：通过环境变量覆盖统一输出根目录。
# 示例：WORK_DIR_OVERRIDE=exp_template/work_dir ./exp_template/train_all_models.sh
WORK_DIR_OVERRIDE="${WORK_DIR_OVERRIDE:-}"

CONFIG_FILES=(
  "${SCRIPT_DIR}/config/resnet_fcn.yaml"
  "${SCRIPT_DIR}/config/unet.yaml"
  "${SCRIPT_DIR}/config/deeplabv3plus.yaml"
  "${SCRIPT_DIR}/config/pspnet.yaml"
  "${SCRIPT_DIR}/config/swin_uperlite.yaml"
)

for cfg in "${CONFIG_FILES[@]}"; do
  if [[ ! -f "${cfg}" ]]; then
    echo "[ERROR] 配置文件不存在: ${cfg}" >&2
    exit 1
  fi
done

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[ERROR] 找不到 Python 解释器: ${PYTHON_BIN}" >&2
  exit 1
fi

ts="$(date +%Y%m%d_%H%M%S)"
log_root="${SCRIPT_DIR}/work_dir/_batch_logs/${ts}"
mkdir -p "${log_root}"

echo "[INFO] 批训练开始: $(date '+%F %T')"
echo "[INFO] Python: ${PYTHON_BIN}"
echo "[INFO] 日志目录: ${log_root}"

fail_count=0
failed_models=()
failed_logs=()

for cfg in "${CONFIG_FILES[@]}"; do
  model_name="$(basename "${cfg}" .yaml)"
  log_file="${log_root}/${model_name}.log"

  cmd=("${PYTHON_BIN}" "${SCRIPT_DIR}/train.py" "--config-file" "${cfg}")
  if [[ -n "${WORK_DIR_OVERRIDE}" ]]; then
    cmd+=("--work-dir" "${WORK_DIR_OVERRIDE}")
  fi

  echo "[INFO] ===== 开始训练 ${model_name} ====="
  echo "[INFO] 命令: ${cmd[*]}"

  # 将每个模型训练日志独立落盘，方便定位失败原因。
  if "${cmd[@]}" 2>&1 | tee "${log_file}"; then
    echo "[INFO] ===== ${model_name} 训练完成 ====="
  else
    echo "[ERROR] ===== ${model_name} 训练失败，日志: ${log_file} =====" >&2
    fail_count=$((fail_count + 1))
    failed_models+=("${model_name}")
    failed_logs+=("${log_file}")
    continue
  fi
done

echo "[INFO] 批训练结束: $(date '+%F %T')"
if [[ ${fail_count} -gt 0 ]]; then
  echo "[ERROR] 共有 ${fail_count} 个模型训练失败：" >&2
  for i in "${!failed_models[@]}"; do
    echo "[ERROR] - ${failed_models[$i]} -> ${failed_logs[$i]}" >&2
  done
  exit 1
fi
echo "[INFO] 全部模型训练成功。"
