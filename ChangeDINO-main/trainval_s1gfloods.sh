#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

DINO_ARCH="${DINO_ARCH:-dinov3_vits16}"
DINO_WEIGHT="${DINO_WEIGHT:-dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth}"
BACKBONE="${BACKBONE:-convnextv2_nano}"
BACKBONE_WEIGHT="${BACKBONE_WEIGHT:-pretrained/convnextv2_nano_22k_224_ema.pt}"
RUN_NAME="${RUN_NAME:-S1GFloods-ChangeDINO-${DINO_ARCH#dinov3_}}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TOPO_GRID="${TOPO_GRID:-16}"
TOPO_K="${TOPO_K:-12}"

cmd=(
python trainval.py \
  --name "${RUN_NAME}" \
  --dataset S1GFloods_CD_DINO \
  --dataroot ../datasets \
  --dataset_mode sar \
  --stats_file ../datasets/S1GFloods_CD_DINO/channel_stats_s1gfloods_train.json \
  --backbone "${BACKBONE}" \
  --dino_arch "${DINO_ARCH}" \
  --dino_weight "${DINO_WEIGHT}" \
  --topo_grid_size "${TOPO_GRID}" \
  --topo_neighbor_k "${TOPO_K}" \
  --gpu_ids 0 \
  --batch_size "${BATCH_SIZE}" \
  --num_epochs 80 \
  --lr 1e-4 \
)

if [[ -n "${BACKBONE_WEIGHT}" ]]; then
  cmd+=(--backbone_weight "${BACKBONE_WEIGHT}")
fi

cmd+=("$@")
"${cmd[@]}"
