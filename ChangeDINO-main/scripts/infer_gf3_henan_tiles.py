#!/usr/bin/env python3
"""GF3 河南整景 SAR 变化检测推理包装脚本。

实际推理逻辑在 `infer_sar_scene_tiles.py`：与训练一致，从 checkpoint 的
`meta.model_config` 读取结构（含 contrast_pool_sizes、micro_gate、topo_grid_size 等）。
当前默认推荐使用稳定三模块版或在其基础上开启 `--micro_gate` 训练得到的权重。

可选：若训练时改过参数且元数据缺失，可显式传入
`--contrast_pool_sizes`、`--micro_gate`、`--topo_grid_size`、`--topo_neighbor_k` 等（与
`option.py` 一致）。
"""

from __future__ import annotations

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from infer_sar_scene_tiles import main as infer_main

# 示例：将 CHECKPOINT 换为你在 trainval_s1gfloods.sh / run.md 下实际训练目录
# 中的 *_efficientnet_b0_best.pth；OUTPUT 建议带实验名与日期便于对照。
"""
python ChangeDINO-main/scripts/infer_gf3_henan_tiles.py \
  --tiles-root datasets/GF3_Henan_CD_infer \
  --checkpoint ChangeDINO-main/checkpoints/S1GFloods-topo-micro-b8-20260409/S1GFloods-topo-micro-b8-20260409_efficientnet_b0_best.pth \
  --stats_file datasets/S1GFloods_CD_DINO/channel_stats_s1gfloods_train.json \
  --gpu_ids 0 \
  --batch_size 8 \
  --output-dir ChangeDINO-main/outputs/gf3_henan_topo_b8

python ChangeDINO-main/scripts/infer_gf3_henan_tiles.py \
  --tiles-root datasets/GF3_Zhuozhou_CD_infer \
  --checkpoint ChangeDINO-main/checkpoints/S1GFloods-topo-b8-YYYYMMDD/S1GFloods-topo-b8-YYYYMMDD_efficientnet_b0_best.pth \
  --stats_file datasets/S1GFloods_CD_DINO/channel_stats_s1gfloods_train.json \
  --gpu_ids 0 \
  --batch_size 8 \
  --output-dir ChangeDINO-main/outputs/gf3_zhuozhou_topo_b8

python ChangeDINO-main/scripts/infer_gf3_henan_tiles.py \
  --tiles-root datasets/GF3_Henan_CD_infer \
  --checkpoint ChangeDINO-main/checkpoints/S1GFloods-hybrid-mv2-b6-20260409-1/S1GFloods-hybrid-mv2-b6-20260409-1_efficientnet_b0_best.pth \
  --stats_file datasets/S1GFloods_CD_DINO/channel_stats_s1gfloods_train.json \
  --gpu_ids 0 \
  --batch_size 8 \
  --output-dir ChangeDINO-main/outputs/gf3_henan_topo_micro_b8

"""


def main() -> None:
    infer_main(
        defaults={
            "tiles_root": "datasets/GF3_Henan_CD_infer",
            "output_dir": "ChangeDINO-main/outputs/gf3_henan",
        },
        description="Infer ChangeDINO on GF3 Henan SAR tiles",
    )


if __name__ == "__main__":
    main()
