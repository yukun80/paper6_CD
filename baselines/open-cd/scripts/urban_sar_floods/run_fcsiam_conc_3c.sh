#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENCD_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_CONFIG="configs/fcsn/fc_siam_diff_512x512_60k_urban_sar_floods_3c.py"
DEFAULT_WORKDIR="work_dirs/fcsiam_diff_urban3c_512x512_60k"
DEFAULT_DATA_ROOT="${OPENCD_DIR}/../../datasets/urban_sar_floods_CD"

MODE="${1:-}"
if [[ -z "${MODE}" ]]; then
  echo "Usage: $0 <check-env|smoke-train|full-train|eval|test> [options]"
  echo "Options:"
  echo "  --save-best <metric>   default: val/mIoU"
  echo "  --best-fscore          shortcut of --save-best val/mFscore"
  exit 1
fi
shift || true

CONFIG="${DEFAULT_CONFIG}"
WORK_DIR="${DEFAULT_WORKDIR}"
DATA_ROOT="${DEFAULT_DATA_ROOT}"
GPUS=1
CHECKPOINT=""
SAVE_BEST="val/mIoU"
CALLER_PWD="$(pwd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"; shift 2;;
    --work-dir)
      WORK_DIR="$2"; shift 2;;
    --data-root)
      DATA_ROOT="$2"; shift 2;;
    --gpus)
      GPUS="$2"; shift 2;;
    --checkpoint)
      CHECKPOINT="$2"; shift 2;;
    --save-best)
      SAVE_BEST="$2"; shift 2;;
    --best-fscore)
      SAVE_BEST="val/mFscore"; shift 1;;
    *)
      echo "Unknown option: $1"
      exit 1;;
  esac
done

if [[ "${DATA_ROOT}" != /* ]]; then
  DATA_ROOT="$(cd "${CALLER_PWD}" && cd "${DATA_ROOT}" 2>/dev/null && pwd || true)"
fi

cd "${OPENCD_DIR}"
export NO_ALBUMENTATIONS_UPDATE=1

if [[ "${MODE}" != "check-env" && ! -d "${DATA_ROOT}" ]]; then
  echo "Invalid data root: ${DATA_ROOT}" >&2
  echo "Use --data-root to set a valid dataset path." >&2
  exit 1
fi

resolve_ckpt() {
  local work_dir="$1"
  local ckpt=""
  local metric_tag="${SAVE_BEST//\//_}"
  ckpt="$(ls -1 "${work_dir}"/best_"${metric_tag}"*.pth 2>/dev/null | head -n 1 || true)"
  if [[ -z "${ckpt}" ]]; then
    ckpt="$(ls -1 "${work_dir}"/best_*.pth 2>/dev/null | head -n 1 || true)"
  fi
  if [[ -z "${ckpt}" && -f "${work_dir}/latest.pth" ]]; then
    ckpt="${work_dir}/latest.pth"
  fi
  if [[ -z "${ckpt}" ]]; then
    echo "No checkpoint found in ${work_dir}" >&2
    exit 1
  fi
  echo "${ckpt}"
}

run_train() {
  local extra_cfg_opts="${1:-}"
  if [[ "${GPUS}" -gt 1 ]]; then
    torchrun --nproc_per_node="${GPUS}" tools/train.py "${CONFIG}" \
      --launcher pytorch \
      --work-dir "${WORK_DIR}" \
      --cfg-options "train_dataloader.dataset.data_root=${DATA_ROOT}" \
                    "val_dataloader.dataset.data_root=${DATA_ROOT}" \
                    "test_dataloader.dataset.data_root=${DATA_ROOT}" \
                    "default_hooks.checkpoint.save_best=${SAVE_BEST}" \
                    "default_hooks.checkpoint.rule=greater" \
                    ${extra_cfg_opts}
  else
    python tools/train.py "${CONFIG}" \
      --work-dir "${WORK_DIR}" \
      --cfg-options "train_dataloader.dataset.data_root=${DATA_ROOT}" \
                    "val_dataloader.dataset.data_root=${DATA_ROOT}" \
                    "test_dataloader.dataset.data_root=${DATA_ROOT}" \
                    "default_hooks.checkpoint.save_best=${SAVE_BEST}" \
                    "default_hooks.checkpoint.rule=greater" \
                    ${extra_cfg_opts}
  fi
}

run_test() {
  local ckpt="${1}"
  if [[ "${GPUS}" -gt 1 ]]; then
    torchrun --nproc_per_node="${GPUS}" tools/test.py "${CONFIG}" "${ckpt}" \
      --launcher pytorch \
      --cfg-options "val_dataloader.dataset.data_root=${DATA_ROOT}" \
                    "test_dataloader.dataset.data_root=${DATA_ROOT}"
  else
    python tools/test.py "${CONFIG}" "${ckpt}" \
      --cfg-options "val_dataloader.dataset.data_root=${DATA_ROOT}" \
                    "test_dataloader.dataset.data_root=${DATA_ROOT}"
  fi
}

case "${MODE}" in
  check-env)
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
        'Please install a compatible 4.x version. '
        f'Import error: {type(e).__name__}: {e}')
print('Version compatibility check passed.')
PY
    ;;
  smoke-train)
    run_train "train_cfg.max_iters=1000 train_cfg.val_interval=200 default_hooks.checkpoint.interval=200"
    ;;
  full-train)
    run_train ""
    ;;
  eval|test)
    if [[ -z "${CHECKPOINT}" ]]; then
      CHECKPOINT="$(resolve_ckpt "${WORK_DIR}")"
    fi
    run_test "${CHECKPOINT}"
    ;;
  *)
    echo "Unsupported mode: ${MODE}"
    exit 1
    ;;
esac
