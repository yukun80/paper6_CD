#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENCD_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_DATA_ROOT="${OPENCD_DIR}/../../datasets/S1GFloods_CD_DINO"
DEFAULT_BATCH_ROOT="${OPENCD_DIR}/work_dirs"
DEFAULT_GPUS=1
DEFAULT_SAVE_BEST="mIoU"
CALLER_PWD="$(pwd)"

CONFIGS=(
  "configs/fcsn/fc_siam_diff_256x256_40k_s1gfloods.py"
  "configs/ifn/ifn_256x256_40k_s1gfloods.py"
  "configs/bit/bit_r18_256x256_40k_s1gfloods.py"
  "configs/changer/changer_ex_r18_256x256_40k_s1gfloods.py"
  "configs/changestar/changestar_farseg_1x96_256x256_40k_s1gfloods.py"
  "configs/lightcdnet/lightcdnet_s_256x256_40k_s1gfloods.py"
)

usage() {
  cat <<'EOF'
Usage: run_all_s1gfloods.sh <check-env|smoke-train|full-train> [options]

Options:
  --data-root <path>   dataset root, default: ../../datasets/S1GFloods_CD_DINO
  --batch-root <path>  parent directory of this batch run, default: ./work_dirs
  --gpus <n>           GPU count for one training job, default: 1
  --save-best <metric> checkpoint metric, default: mIoU
EOF
}

MODE="${1:-}"
if [[ -z "${MODE}" ]]; then
  usage
  exit 1
fi
shift || true

DATA_ROOT="${DEFAULT_DATA_ROOT}"
BATCH_ROOT="${DEFAULT_BATCH_ROOT}"
GPUS="${DEFAULT_GPUS}"
SAVE_BEST="${DEFAULT_SAVE_BEST}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root)
      DATA_ROOT="$2"; shift 2;;
    --batch-root)
      BATCH_ROOT="$2"; shift 2;;
    --gpus)
      GPUS="$2"; shift 2;;
    --save-best)
      SAVE_BEST="$2"; shift 2;;
    -h|--help)
      usage
      exit 0;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1;;
  esac
done

resolve_path() {
  local raw_path="$1"
  python - "${CALLER_PWD}" "${raw_path}" <<'PY'
import os
import sys

caller_pwd = sys.argv[1]
raw_path = sys.argv[2]
if os.path.isabs(raw_path):
    print(os.path.abspath(raw_path))
else:
    print(os.path.abspath(os.path.join(caller_pwd, raw_path)))
PY
}

DATA_ROOT="$(resolve_path "${DATA_ROOT}")"
BATCH_ROOT="$(resolve_path "${BATCH_ROOT}")"

if [[ -z "${BATCH_ROOT}" ]]; then
  echo "Failed to resolve batch root." >&2
  exit 1
fi

resolve_unique_batch_dir() {
  local parent_dir="$1"
  local base_name="$2"
  local candidate="${parent_dir}/${base_name}"
  local index=1

  while [[ -e "${candidate}" ]]; do
    candidate="${parent_dir}/${base_name}-${index}"
    index=$((index + 1))
  done
  printf '%s\n' "${candidate}"
}

resolve_ckpt() {
  local work_dir="$1"
  local metric_tag="${SAVE_BEST//\//_}"
  local ckpt=""

  ckpt="$(ls -1 "${work_dir}"/best_"${metric_tag}"*.pth 2>/dev/null | head -n 1 || true)"
  if [[ -z "${ckpt}" ]]; then
    ckpt="$(ls -1 "${work_dir}"/best_*.pth 2>/dev/null | head -n 1 || true)"
  fi
  if [[ -z "${ckpt}" && -f "${work_dir}/latest.pth" ]]; then
    ckpt="${work_dir}/latest.pth"
  fi
  printf '%s\n' "${ckpt}"
}

check_env() {
  if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "Invalid data root: ${DATA_ROOT}" >&2
    exit 1
  fi

  cd "${OPENCD_DIR}"
  export NO_ALBUMENTATIONS_UPDATE=1
  export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_opencd_s1gfloods}"
  mkdir -p "${MPLCONFIGDIR}"

  for config in "${CONFIGS[@]}"; do
    if [[ ! -f "${OPENCD_DIR}/${config}" ]]; then
      echo "Missing config: ${config}" >&2
      exit 1
    fi
  done

  python - <<'PY'
import importlib
from mmengine.utils import digit_version

mods = ['mmcv', 'mmseg', 'mmengine', 'opencd', 'torch']
vers = {}
for m in mods:
    mod = importlib.import_module(m)
    vers[m] = getattr(mod, '__version__', 'unknown')
    print(f'{m}=={vers[m]}')

mmcv_v = digit_version(vers['mmcv'])
mmseg_v = digit_version(vers['mmseg'])
if not (digit_version('2.0.0rc4') <= mmcv_v < digit_version('2.2.0')):
    raise SystemExit(f'Incompatible mmcv version: {vers["mmcv"]}')
if mmseg_v < digit_version('1.2.0'):
    raise SystemExit(f'mmseg should be >=1.2.0, got {vers["mmseg"]}')
try:
    from transformers.modeling_utils import apply_chunking_to_forward  # noqa: F401
except Exception as e:
    raise SystemExit(
        'Incompatible transformers for mmpretrain BLIP import path. '
        f'Import error: {type(e).__name__}: {e}')
print('Version compatibility check passed.')
PY

  echo "Data root: ${DATA_ROOT}"
  echo "Configs checked: ${#CONFIGS[@]}"
}

build_train_cmd() {
  local config="$1"
  local work_dir="$2"
  local mode="$3"
  local extra_cfg_opts=(
    "train_dataloader.dataset.data_root=${DATA_ROOT}"
    "val_dataloader.dataset.data_root=${DATA_ROOT}"
    "test_dataloader.dataset.data_root=${DATA_ROOT}"
    "default_hooks.checkpoint.save_best=${SAVE_BEST}"
    "default_hooks.checkpoint.rule=greater"
  )

  if [[ "${mode}" == "smoke-train" ]]; then
    extra_cfg_opts+=(
      "train_cfg.max_iters=2"
      "train_cfg.val_interval=2"
      "default_hooks.checkpoint.interval=2"
      "train_dataloader.batch_size=2"
      "train_dataloader.num_workers=0"
      "train_dataloader.persistent_workers=False"
      "val_dataloader.num_workers=0"
      "val_dataloader.persistent_workers=False"
      "test_dataloader.num_workers=0"
      "test_dataloader.persistent_workers=False"
    )
  fi

  if [[ "${GPUS}" -gt 1 ]]; then
    printf '%s\0' torchrun --nproc_per_node="${GPUS}" tools/train.py "${config}" \
      --launcher pytorch \
      --work-dir "${work_dir}" \
      --cfg-options "${extra_cfg_opts[@]}"
  else
    printf '%s\0' python tools/train.py "${config}" \
      --work-dir "${work_dir}" \
      --cfg-options "${extra_cfg_opts[@]}"
  fi
}

run_batch_train() {
  if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "Invalid data root: ${DATA_ROOT}" >&2
    exit 1
  fi

  mkdir -p "${BATCH_ROOT}"
  cd "${OPENCD_DIR}"

  export NO_ALBUMENTATIONS_UPDATE=1

  local base_name="s1gfloods-batch-$(date +%Y%m%d-%H%M%S)"
  local batch_dir
  batch_dir="$(resolve_unique_batch_dir "${BATCH_ROOT}" "${base_name}")"
  local log_dir="${batch_dir}/logs"
  local summary_file="${batch_dir}/summary.tsv"
  local success_file="${batch_dir}/succeeded_models.txt"
  local failed_file="${batch_dir}/failed_models.txt"

  mkdir -p "${batch_dir}" "${log_dir}"
  export MPLCONFIGDIR="${batch_dir}/mplconfig"
  mkdir -p "${MPLCONFIGDIR}"

  printf 'model_tag\tstatus\texit_code\tconfig\twork_dir\tbest_ckpt\tlog_file\tstarted_at\tended_at\tduration_sec\n' > "${summary_file}"
  : > "${success_file}"
  : > "${failed_file}"

  local total=0
  local success_count=0
  local failed_count=0

  echo "Batch dir: ${batch_dir}"
  echo "Data root: ${DATA_ROOT}"
  echo "Mode: ${MODE}"
  echo "Models: ${#CONFIGS[@]}"

  for config in "${CONFIGS[@]}"; do
    total=$((total + 1))

    local model_tag
    model_tag="$(basename "${config}" .py)"
    local work_dir="${batch_dir}/${model_tag}"
    local log_file="${log_dir}/${model_tag}.log"
    local best_ckpt=""
    local started_at ended_at duration_sec status_code

    started_at="$(date '+%F %T')"
    local start_epoch
    start_epoch="$(date +%s)"

    echo
    echo "[$total/${#CONFIGS[@]}] Training ${model_tag}"
    echo "Config: ${config}"
    echo "Work dir: ${work_dir}"
    echo "Log file: ${log_file}"

    local cmd=()
    while IFS= read -r -d '' token; do
      cmd+=("${token}")
    done < <(build_train_cmd "${config}" "${work_dir}" "${MODE}")

    set +e
    "${cmd[@]}" 2>&1 | tee "${log_file}"
    status_code=${PIPESTATUS[0]}
    set -e

    ended_at="$(date '+%F %T')"
    duration_sec="$(( $(date +%s) - start_epoch ))"

    if [[ "${status_code}" -eq 0 ]]; then
      best_ckpt="$(resolve_ckpt "${work_dir}")"
      printf '%s\n' "${model_tag}" >> "${success_file}"
      printf '%s\tsuccess\t0\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${model_tag}" "${config}" "${work_dir}" "${best_ckpt}" "${log_file}" \
        "${started_at}" "${ended_at}" "${duration_sec}" >> "${summary_file}"
      success_count=$((success_count + 1))
      echo "Completed ${model_tag}"
      if [[ -n "${best_ckpt}" ]]; then
        echo "Best checkpoint: ${best_ckpt}"
      fi
    else
      printf '%s\n' "${model_tag}" >> "${failed_file}"
      printf '%s\tfailed\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${model_tag}" "${status_code}" "${config}" "${work_dir}" "" "${log_file}" \
        "${started_at}" "${ended_at}" "${duration_sec}" >> "${summary_file}"
      failed_count=$((failed_count + 1))
      echo "Failed ${model_tag} with exit code ${status_code}" >&2
    fi
  done

  echo
  echo "Batch finished."
  echo "Succeeded: ${success_count}"
  echo "Failed: ${failed_count}"
  echo "Summary: ${summary_file}"
  echo "Success list: ${success_file}"
  echo "Failure list: ${failed_file}"

  if [[ "${failed_count}" -gt 0 ]]; then
    exit 1
  fi
}

case "${MODE}" in
  check-env)
    check_env
    ;;
  smoke-train|full-train)
    run_batch_train
    ;;
  *)
    echo "Unsupported mode: ${MODE}" >&2
    usage
    exit 1
    ;;
esac
