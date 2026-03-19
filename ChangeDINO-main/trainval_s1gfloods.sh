#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python trainval.py \
  --name S1GFloods-ChangeDINO \
  --dataset S1GFloods_CD_DINO \
  --dataroot ../datasets \
  --dataset_mode sar \
  --stats_file ../datasets/S1GFloods_CD_DINO/channel_stats_s1gfloods_train.json \
  --gpu_ids 0 \
  --batch_size 8 \
  --num_epochs 100 \
  --lr 1e-4
