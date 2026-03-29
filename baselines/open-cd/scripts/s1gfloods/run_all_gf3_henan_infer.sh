#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENCD_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TOOLS_DIR="${OPENCD_DIR}/tools"
DEFAULT_WORKDIR_ROOT="${OPENCD_DIR}/work_dirs"
DEFAULT_DATA_ROOT="${OPENCD_DIR}/../../datasets/GF3_Henan_CD_infer"
DEFAULT_DEVICE="cuda:0"
DEFAULT_BATCH_SIZE=4
DEFAULT_THRESHOLD=0.5
CALLER_PWD="$(pwd)"

usage() {
  cat <<'EOF'
Usage: run_all_gf3_henan_infer.sh [options]

Options:
  --batch-dir <path>      batch directory containing trained model work_dirs
  --workdir-root <path>   parent directory of s1gfloods batches, default: ./work_dirs
  --data-root <path>      GF3 infer dataset root, default: ../../datasets/GF3_Henan_CD_infer
  --device <device>       inference device, default: cuda:0
  --batch-size <n>        inference batch size for one model, default: 4
  --threshold <x>         threshold on stitched probability map, default: 0.5
  --limit <n>             optional cap on tiles per model, default: all
  --skip-mosaic           only save tile-level PNG predictions
  -h, --help              show this help
EOF
}

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

resolve_latest_batch_dir() {
  local workdir_root="$1"
  local latest_dir

  latest_dir="$(
    find "${workdir_root}" -maxdepth 1 -mindepth 1 -type d -name 's1gfloods-batch-*' | sort | tail -n 1
  )"
  if [[ -z "${latest_dir}" ]]; then
    echo "No s1gfloods batch directory found under ${workdir_root}" >&2
    exit 1
  fi
  printf '%s\n' "${latest_dir}"
}

resolve_checkpoint() {
  local work_dir="$1"
  local ckpt=""

  ckpt="$(find "${work_dir}" -maxdepth 1 -type f -name 'best_*.pth' | sort | head -n 1 || true)"
  if [[ -z "${ckpt}" && -f "${work_dir}/latest.pth" ]]; then
    ckpt="${work_dir}/latest.pth"
  fi
  if [[ -z "${ckpt}" ]]; then
    echo "No checkpoint found in ${work_dir}" >&2
    return 1
  fi
  printf '%s\n' "${ckpt}"
}

resolve_config() {
  local work_dir="$1"
  local model_tag
  local config_path=""

  model_tag="$(basename "${work_dir}")"
  if [[ -f "${work_dir}/${model_tag}.py" ]]; then
    config_path="${work_dir}/${model_tag}.py"
  fi
  if [[ -z "${config_path}" ]]; then
    echo "Config snapshot not found for ${work_dir}" >&2
    return 1
  fi
  printf '%s\n' "${config_path}"
}

BATCH_DIR=""
WORKDIR_ROOT="${DEFAULT_WORKDIR_ROOT}"
DATA_ROOT="${DEFAULT_DATA_ROOT}"
DEVICE="${DEFAULT_DEVICE}"
BATCH_SIZE="${DEFAULT_BATCH_SIZE}"
THRESHOLD="${DEFAULT_THRESHOLD}"
LIMIT=0
SKIP_MOSAIC=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --batch-dir)
      BATCH_DIR="$2"; shift 2;;
    --workdir-root)
      WORKDIR_ROOT="$2"; shift 2;;
    --data-root)
      DATA_ROOT="$2"; shift 2;;
    --device)
      DEVICE="$2"; shift 2;;
    --batch-size)
      BATCH_SIZE="$2"; shift 2;;
    --threshold)
      THRESHOLD="$2"; shift 2;;
    --limit)
      LIMIT="$2"; shift 2;;
    --skip-mosaic)
      SKIP_MOSAIC=1; shift 1;;
    -h|--help)
      usage
      exit 0;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1;;
  esac
done

WORKDIR_ROOT="$(resolve_path "${WORKDIR_ROOT}")"
DATA_ROOT="$(resolve_path "${DATA_ROOT}")"

if [[ -n "${BATCH_DIR}" ]]; then
  BATCH_DIR="$(resolve_path "${BATCH_DIR}")"
else
  BATCH_DIR="$(resolve_latest_batch_dir "${WORKDIR_ROOT}")"
fi

if [[ ! -d "${BATCH_DIR}" ]]; then
  echo "Invalid batch dir: ${BATCH_DIR}" >&2
  exit 1
fi
if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "Invalid data root: ${DATA_ROOT}" >&2
  exit 1
fi
if ! [[ "${BATCH_SIZE}" =~ ^[0-9]+$ ]] || [[ "${BATCH_SIZE}" -le 0 ]]; then
  echo "Invalid --batch-size: ${BATCH_SIZE}" >&2
  exit 1
fi
if ! [[ "${LIMIT}" =~ ^[0-9]+$ ]] || [[ "${LIMIT}" -lt 0 ]]; then
  echo "Invalid --limit: ${LIMIT}" >&2
  exit 1
fi
if ! python - "${THRESHOLD}" <<'PY'
import sys
value = float(sys.argv[1])
raise SystemExit(0 if 0.0 <= value <= 1.0 else 1)
PY
then
  echo "Invalid --threshold: ${THRESHOLD}" >&2
  exit 1
fi

cd "${OPENCD_DIR}"
export NO_ALBUMENTATIONS_UPDATE=1

LOG_DIR="${BATCH_DIR}/infer_logs"
SUMMARY_FILE="${BATCH_DIR}/infer_gf3_henan_summary.tsv"
SUCCESS_FILE="${BATCH_DIR}/succeeded_models.txt"
mkdir -p "${LOG_DIR}"
printf 'model_tag\tstatus\tconfig\tcheckpoint\ttile_output_dir\tmosaic_dir\treport_file\tlog_file\n' > "${SUMMARY_FILE}"

if [[ -f "${SUCCESS_FILE}" ]]; then
  mapfile -t MODEL_TAGS < "${SUCCESS_FILE}"
else
  mapfile -t MODEL_TAGS < <(
    find "${BATCH_DIR}" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | sort
  )
fi

echo "Batch dir: ${BATCH_DIR}"
echo "Data root: ${DATA_ROOT}"
echo "Device: ${DEVICE}"
echo "Model count: ${#MODEL_TAGS[@]}"

success_count=0
failed_count=0

for model_tag in "${MODEL_TAGS[@]}"; do
  [[ -z "${model_tag}" ]] && continue

  work_dir="${BATCH_DIR}/${model_tag}"
  if [[ ! -d "${work_dir}" ]]; then
    echo "Skip missing work dir: ${work_dir}" >&2
    continue
  fi

  log_file="${LOG_DIR}/${model_tag}.log"
  out_dir="${work_dir}/infer_gf3_henan_png"
  mosaic_dir="${work_dir}/infer_gf3_henan_full"
  report_file="${mosaic_dir}/infer_report.json"

  echo
  echo "Infer model: ${model_tag}"
  echo "Work dir: ${work_dir}"
  echo "Log file: ${log_file}"

  set +e
  config_path="$(resolve_config "${work_dir}")"
  config_status=$?
  set -e
  if [[ "${config_status}" -ne 0 ]]; then
    printf '%s\tfailed\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${model_tag}" "" "" "${out_dir}" "${mosaic_dir}" "${report_file}" "${log_file}" >> "${SUMMARY_FILE}"
    failed_count=$((failed_count + 1))
    continue
  fi

  set +e
  checkpoint_path="$(resolve_checkpoint "${work_dir}")"
  checkpoint_status=$?
  set -e
  if [[ "${checkpoint_status}" -ne 0 ]]; then
    printf '%s\tfailed\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${model_tag}" "${config_path}" "" "${out_dir}" "${mosaic_dir}" "${report_file}" "${log_file}" >> "${SUMMARY_FILE}"
    failed_count=$((failed_count + 1))
    continue
  fi

  cmd=(
    python "${TOOLS_DIR}/infer_gf3_henan.py"
    "${config_path}"
    "${checkpoint_path}"
    --data-root "${DATA_ROOT}"
    --work-dir "${work_dir}"
    --out-dir "${out_dir}"
    --mosaic-dir "${mosaic_dir}"
    --device "${DEVICE}"
    --batch-size "${BATCH_SIZE}"
    --threshold "${THRESHOLD}"
  )
  if [[ "${LIMIT}" -gt 0 ]]; then
    cmd+=(--limit "${LIMIT}")
  fi
  if [[ "${SKIP_MOSAIC}" -eq 1 ]]; then
    cmd+=(--skip-mosaic)
    mosaic_dir=""
    report_file=""
  fi

  set +e
  "${cmd[@]}" 2>&1 | tee "${log_file}"
  status_code=${PIPESTATUS[0]}
  set -e

  if [[ "${status_code}" -eq 0 ]]; then
    printf '%s\tsuccess\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${model_tag}" "${config_path}" "${checkpoint_path}" "${out_dir}" "${mosaic_dir}" "${report_file}" "${log_file}" >> "${SUMMARY_FILE}"
    success_count=$((success_count + 1))
  else
    printf '%s\tfailed\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${model_tag}" "${config_path}" "${checkpoint_path}" "${out_dir}" "${mosaic_dir}" "${report_file}" "${log_file}" >> "${SUMMARY_FILE}"
    failed_count=$((failed_count + 1))
  fi
done

echo
echo "Inference finished."
echo "Succeeded: ${success_count}"
echo "Failed: ${failed_count}"
echo "Summary: ${SUMMARY_FILE}"

if [[ "${failed_count}" -gt 0 ]]; then
  exit 1
fi
