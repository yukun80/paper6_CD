#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python trainval.py \
  --name ETCI2021-ChangeDINO \
  --dataset ETCI2021_CD_DINO \
  --dataroot ../datasets \
  --dataset_mode sar \
  --stats_file ../datasets/ETCI2021_CD_DINO/channel_stats_etci2021_train.json \
  --gpu_ids 0 \
  --batch_size 8 \
  --num_epochs 100 \
  --lr 1e-4
