#!/usr/bin/env bash
set -euo pipefail

# 一次性顺序执行 exp_template 全部模型推理可视化（默认验证集）。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

SPLIT="${SPLIT:-val}"
SAVE_EXT="${SAVE_EXT:-jpg}"
JPG_QUALITY="${JPG_QUALITY:-95}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"
OUT_ROOT="${OUT_ROOT:-}"

MODELS=(
  "resnet_fcn|${SCRIPT_DIR}/config/resnet_fcn.yaml|${SCRIPT_DIR}/work_dir/resnet_fcn_resnet50_20260306_224305/checkpoints/best.pth"
  "unet|${SCRIPT_DIR}/config/unet.yaml|${SCRIPT_DIR}/work_dir/unet_unet_20260306_235046/checkpoints/best.pth"
  "deeplabv3plus|${SCRIPT_DIR}/config/deeplabv3plus.yaml|${SCRIPT_DIR}/work_dir/deeplabv3plus_resnet50_20260307_011017/checkpoints/best.pth"
  "pspnet|${SCRIPT_DIR}/config/pspnet.yaml|${SCRIPT_DIR}/work_dir/pspnet_resnet50_20260307_163549/checkpoints/best.pth"
  "swin_uperlite|${SCRIPT_DIR}/config/swin_uperlite.yaml|${SCRIPT_DIR}/work_dir/swin_uperlite_swin_b_20260307_022443/checkpoints/best.pth"
)

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[ERROR] 找不到 Python 解释器: ${PYTHON_BIN}" >&2
  exit 1
fi

ts="$(date +%Y%m%d_%H%M%S)"
log_root="${SCRIPT_DIR}/work_dir/_batch_vis_logs/${ts}"
mkdir -p "${log_root}"

echo "[INFO] 批量可视化开始: $(date '+%F %T')"
echo "[INFO] Python: ${PYTHON_BIN}"
echo "[INFO] split=${SPLIT}, save_ext=${SAVE_EXT}, jpg_quality=${JPG_QUALITY}, max_samples=${MAX_SAMPLES}"
echo "[INFO] 日志目录: ${log_root}"

fail_count=0
failed_models=()
failed_logs=()

for entry in "${MODELS[@]}"; do
  IFS='|' read -r model_name cfg ckpt <<<"${entry}"
  log_file="${log_root}/${model_name}.log"

  if [[ ! -f "${cfg}" ]]; then
    echo "[ERROR] 配置文件不存在: ${cfg}" >&2
    exit 1
  fi
  if [[ ! -f "${ckpt}" ]]; then
    echo "[ERROR] checkpoint 不存在: ${ckpt}" >&2
    exit 1
  fi

  cmd=(
    "${PYTHON_BIN}" "${SCRIPT_DIR}/visualize.py"
    "--config-file" "${cfg}"
    "--ckpt" "${ckpt}"
    "--split" "${SPLIT}"
    "--save-ext" "${SAVE_EXT}"
    "--jpg-quality" "${JPG_QUALITY}"
    "--max-samples" "${MAX_SAMPLES}"
  )

  if [[ -n "${OUT_ROOT}" ]]; then
    out_dir="${OUT_ROOT}/${model_name}/${SPLIT}"
    cmd+=("--out-dir" "${out_dir}")
  fi

  echo "[INFO] ===== 开始可视化 ${model_name} ====="
  echo "[INFO] 命令: ${cmd[*]}"

  if "${cmd[@]}" 2>&1 | tee "${log_file}"; then
    echo "[INFO] ===== ${model_name} 可视化完成 ====="
  else
    echo "[ERROR] ===== ${model_name} 可视化失败，日志: ${log_file} =====" >&2
    fail_count=$((fail_count + 1))
    failed_models+=("${model_name}")
    failed_logs+=("${log_file}")
    continue
  fi
done

echo "[INFO] 批量可视化结束: $(date '+%F %T')"
if [[ ${fail_count} -gt 0 ]]; then
  echo "[ERROR] 共有 ${fail_count} 个模型可视化失败：" >&2
  for i in "${!failed_models[@]}"; do
    echo "[ERROR] - ${failed_models[$i]} -> ${failed_logs[$i]}" >&2
  done
  exit 1
fi
echo "[INFO] 全部模型可视化成功。"
