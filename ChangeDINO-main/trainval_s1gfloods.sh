#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

DINO_ARCH="${DINO_ARCH:-dinov3_vits16}"
DINO_WEIGHT="${DINO_WEIGHT:-dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth}"
BACKBONE="${BACKBONE:-efficientnet_b0}"
BACKBONE_WEIGHT="${BACKBONE_WEIGHT:-pretrained/efficientnet_b0_ra-3dd342df.pth}"
RUN_NAME="${RUN_NAME:-S1GFloods-ChangeDINO-${DINO_ARCH#dinov3_}}"
BATCH_SIZE="${BATCH_SIZE:-6}"
TOPO_GRID="${TOPO_GRID:-16}"
TOPO_K="${TOPO_K:-12}"
TOPO_NEIGHBOR_MODE="${TOPO_NEIGHBOR_MODE:-mixed}"
TOPO_LONG_OFFSETS="${TOPO_LONG_OFFSETS:-2 4}"
TOPO_HOPS="${TOPO_HOPS:-3}"
TOPO_MIN_NODE_OCC="${TOPO_MIN_NODE_OCC:-0.25}"
MICRO_GATE="${MICRO_GATE:-1}"
REFINER="${REFINER:-hybrid}"
DINO_COLLAB_MODE="${DINO_COLLAB_MODE:-multilevel_v2}"
BRANCH_CONSISTENCY_WEIGHT="${BRANCH_CONSISTENCY_WEIGHT:-0.05}"
CONSISTENCY_WARMUP_EPOCHS="${CONSISTENCY_WARMUP_EPOCHS:-15}"

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
  --topo_neighbor_mode "${TOPO_NEIGHBOR_MODE}" \
  --topo_long_offsets ${TOPO_LONG_OFFSETS} \
  --topo_n_hops "${TOPO_HOPS}" \
  --topo_min_node_occ "${TOPO_MIN_NODE_OCC}" \
  --refiner "${REFINER}" \
  --dino_collab_mode "${DINO_COLLAB_MODE}" \
  --branch_consistency_weight "${BRANCH_CONSISTENCY_WEIGHT}" \
  --consistency_warmup_epochs "${CONSISTENCY_WARMUP_EPOCHS}" \
  --gpu_ids 0 \
  --batch_size "${BATCH_SIZE}" \
  --num_epochs 80 \
  --lr 1e-4 \
)

if [[ -n "${BACKBONE_WEIGHT}" ]]; then
  cmd+=(--backbone_weight "${BACKBONE_WEIGHT}")
fi

if [[ "${MICRO_GATE}" == "1" ]]; then
  cmd+=(--micro_gate)
fi

cmd+=("$@")
"${cmd[@]}"
