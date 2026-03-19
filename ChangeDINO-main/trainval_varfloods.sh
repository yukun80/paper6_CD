#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python trainval.py \
  --name VarFloods-ChangeDINO \
  --dataset VarFloods_CD \
  --dataroot ../datasets \
  --dataset_mode sar \
  --stats_file ../datasets/VarFloods_CD/channel_stats_varfloods_train.json \
  --gpu_ids 0 \
  --batch_size 8 \
  --input_size 256 \
  --num_epochs 100 \
  --lr 1e-4
