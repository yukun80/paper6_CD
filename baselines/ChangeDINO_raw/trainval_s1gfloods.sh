#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

DINO_ARCH="${DINO_ARCH:-dinov3_vits16}"
DINO_WEIGHT="${DINO_WEIGHT:-../../ChangeDINO-main/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth}"
RUN_NAME="${RUN_NAME:-S1GFloods-ChangeDINO-${DINO_ARCH#dinov3_}}"
DATA_ROOT="${DATA_ROOT:-../../datasets}"
STATS_FILE="${STATS_FILE:-../../datasets/S1GFloods_CD_DINO/channel_stats_s1gfloods_train.json}"
SAVE_EPOCH_FREQ="${SAVE_EPOCH_FREQ:-10}"

if [[ ! -f "${DINO_WEIGHT}" ]]; then
  echo "Missing DINO weight: ${DINO_WEIGHT}" >&2
  exit 1
fi

if [[ ! -d "${DATA_ROOT}/S1GFloods_CD_DINO" ]]; then
  echo "Missing dataset directory: ${DATA_ROOT}/S1GFloods_CD_DINO" >&2
  exit 1
fi

if [[ ! -f "${STATS_FILE}" ]]; then
  echo "Missing stats file: ${STATS_FILE}" >&2
  exit 1
fi

echo "Run name: ${RUN_NAME}"
echo "DINO weight: ${DINO_WEIGHT}"
echo "Data root: ${DATA_ROOT}"
echo "Stats file: ${STATS_FILE}"
echo "Save epoch freq: ${SAVE_EPOCH_FREQ}"

python trainval.py \
  --name "${RUN_NAME}" \
  --dataset S1GFloods_CD_DINO \
  --dataroot "${DATA_ROOT}" \
  --dataset_mode sar \
  --stats_file "${STATS_FILE}" \
  --dino_arch "${DINO_ARCH}" \
  --dino_weight "${DINO_WEIGHT}" \
  --gpu_ids 0 \
  --batch_size 12 \
  --num_epochs 100 \
  --save_epoch_freq "${SAVE_EPOCH_FREQ}" \
  --lr 1e-4 \
  "$@"
