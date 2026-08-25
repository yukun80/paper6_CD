#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

DINO_ARCH="${DINO_ARCH:-dinov3_vits16}"
DINO_WEIGHT="${DINO_WEIGHT:-dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth}"
BACKBONE_WEIGHT="${BACKBONE_WEIGHT:-pretrained/efficientnet_b2_ra-bcdf34b7.pth}"
RUN_NAME="${RUN_NAME:-S1GFloods-HA-CQI-B2-${DINO_ARCH#dinov3_}}"
DATASET_NAME="${DATASET_NAME:-S1GFloods_CD_DINO_BG_75_25}"
DATA_ROOT="${DATA_ROOT:-../datasets}"
STATS_FILE="${STATS_FILE:-${DATA_ROOT}/${DATASET_NAME}/channel_stats_s1gfloods_train.json}"
BATCH_SIZE="${BATCH_SIZE:-12}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LR="${LR:-1e-4}"
SOFT_ALIGNMENT="${SOFT_ALIGNMENT:-1}"
NUM_CHANGE_QUERIES="${NUM_CHANGE_QUERIES:-16}"
CQI_HEADS="${CQI_HEADS:-4}"
EVAL_FG_THRESHOLD="${EVAL_FG_THRESHOLD:-0.40}"
THRESHOLD_MIN="${THRESHOLD_MIN:-0.05}"
THRESHOLD_MAX="${THRESHOLD_MAX:-0.95}"
THRESHOLD_STEP="${THRESHOLD_STEP:-0.01}"
SEED="${SEED:-1}"
FOCAL_BG_WEIGHT="${FOCAL_BG_WEIGHT:-0.25}"
FOCAL_FG_WEIGHT="${FOCAL_FG_WEIGHT:-0.75}"
RESUME="${RESUME:-}"
HEAD_LR_MULT="${HEAD_LR_MULT:-2.0}"
AUX_LOSS_WEIGHT="${AUX_LOSS_WEIGHT:-1.0}"
AUX_LOSS_WEIGHT_END="${AUX_LOSS_WEIGHT_END:-0.5}"
AUX_DECAY_START_EPOCH="${AUX_DECAY_START_EPOCH:-5}"
TVERSKY_BETA_START="${TVERSKY_BETA_START:-0.70}"
TVERSKY_BETA_END="${TVERSKY_BETA_END:-0.55}"
LOSS_ANNEAL_EPOCHS="${LOSS_ANNEAL_EPOCHS:-20}"
SUPPORT_CONSISTENCY_WEIGHT="${SUPPORT_CONSISTENCY_WEIGHT:-0.03}"
COARSE_CONSISTENCY_WEIGHT="${COARSE_CONSISTENCY_WEIGHT:-0.02}"
CONSISTENCY_WARMUP_EPOCHS="${CONSISTENCY_WARMUP_EPOCHS:-5}"
CONSISTENCY_RAMP_EPOCHS="${CONSISTENCY_RAMP_EPOCHS:-10}"
AMP="${AMP:-1}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"

if [[ ! -d "${DATA_ROOT}/${DATASET_NAME}" ]]; then
  echo "Dataset directory not found: ${DATA_ROOT}/${DATASET_NAME}" >&2
  exit 1
fi

if [[ ! -f "${STATS_FILE}" ]]; then
  echo "Stats file not found: ${STATS_FILE}" >&2
  exit 1
fi

if [[ ! -f "${DINO_WEIGHT}" ]]; then
  echo "DINO weight not found: ${DINO_WEIGHT}" >&2
  exit 1
fi

if [[ -n "${BACKBONE_WEIGHT}" && ! -f "${BACKBONE_WEIGHT}" ]]; then
  echo "Backbone weight not found: ${BACKBONE_WEIGHT}" >&2
  exit 1
fi

cmd=(
python trainval.py \
  --name "${RUN_NAME}" \
  --dataset "${DATASET_NAME}" \
  --dataroot "${DATA_ROOT}" \
  --dataset_mode sar \
  --stats_file "${STATS_FILE}" \
  --dino_arch "${DINO_ARCH}" \
  --dino_weight "${DINO_WEIGHT}" \
  --num_change_queries "${NUM_CHANGE_QUERIES}" \
  --cqi_heads "${CQI_HEADS}" \
  --head_lr_mult "${HEAD_LR_MULT}" \
  --aux_loss_weight "${AUX_LOSS_WEIGHT}" \
  --aux_loss_weight_end "${AUX_LOSS_WEIGHT_END}" \
  --aux_decay_start_epoch "${AUX_DECAY_START_EPOCH}" \
  --tversky_beta_start "${TVERSKY_BETA_START}" \
  --tversky_beta_end "${TVERSKY_BETA_END}" \
  --loss_anneal_epochs "${LOSS_ANNEAL_EPOCHS}" \
  --support_consistency_weight "${SUPPORT_CONSISTENCY_WEIGHT}" \
  --coarse_consistency_weight "${COARSE_CONSISTENCY_WEIGHT}" \
  --consistency_warmup_epochs "${CONSISTENCY_WARMUP_EPOCHS}" \
  --consistency_ramp_epochs "${CONSISTENCY_RAMP_EPOCHS}" \
  --focal_class_weights "${FOCAL_BG_WEIGHT}" "${FOCAL_FG_WEIGHT}" \
  --eval_fg_threshold "${EVAL_FG_THRESHOLD}" \
  --threshold_min "${THRESHOLD_MIN}" \
  --threshold_max "${THRESHOLD_MAX}" \
  --threshold_step "${THRESHOLD_STEP}" \
  --seed "${SEED}" \
  --gpu_ids 0 \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --num_epochs 80 \
  --lr "${LR}" \
)

if [[ -n "${BACKBONE_WEIGHT}" ]]; then
  cmd+=(--backbone_weight "${BACKBONE_WEIGHT}")
fi

if [[ "${SOFT_ALIGNMENT}" != "1" ]]; then
  cmd+=(--disable_soft_alignment)
fi

if [[ "${AMP}" == "1" ]]; then
  cmd+=(--amp --amp_dtype "${AMP_DTYPE}")
elif [[ "${AMP}" == "0" ]]; then
  cmd+=(--no-amp)
else
  echo "AMP must be 0 or 1, got: ${AMP}" >&2
  exit 1
fi

if [[ -n "${RESUME}" ]]; then
  cmd+=(--resume "${RESUME}")
fi

cmd+=("$@")
"${cmd[@]}"
