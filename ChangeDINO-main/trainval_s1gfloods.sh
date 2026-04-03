#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

DINO_ARCH="${DINO_ARCH:-dinov3_vits16}"
DINO_WEIGHT="${DINO_WEIGHT:-dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth}"
BACKBONE="${BACKBONE:-convnextv2_nano}"
BACKBONE_WEIGHT="${BACKBONE_WEIGHT:-pretrained/convnextv2_nano_22k_224_ema.pt}"
RUN_NAME="${RUN_NAME:-S1GFloods-ChangeDINO-${DINO_ARCH#dinov3_}}"
# grid_size=32 高精度模式：batch 降至 4 保留显存余量；如需恢复低精度可覆盖环境变量
BATCH_SIZE="${BATCH_SIZE:-4}"
TOPO_GRID="${TOPO_GRID:-32}"
TOPO_K="${TOPO_K:-16}"

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
  --gpu_ids 0 \
  --batch_size "${BATCH_SIZE}" \
  --num_epochs 100 \
  --lr 1e-4 \
  --topo_grid_size "${TOPO_GRID}" \
  --topo_neighbor_k "${TOPO_K}" \
)

if [[ -n "${BACKBONE_WEIGHT}" ]]; then
  cmd+=(--backbone_weight "${BACKBONE_WEIGHT}")
fi

cmd+=("$@")
"${cmd[@]}"
