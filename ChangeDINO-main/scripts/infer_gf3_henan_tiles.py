#!/usr/bin/env python3
"""GF3 河南整景 SAR 变化检测推理包装脚本。

实际推理逻辑在 `infer_sar_scene_tiles.py`：与训练一致，从 checkpoint 的
`meta.model_config` 读取结构（含 contrast_pool_size、topo_grid_size 等）。
请使用 **模块 2/3 改进后**（ContrastAwareDiff + FloodTopoRouter 双向精修版）
训练得到的权重。

可选：若训练时改过参数且元数据缺失，可显式传入
`--contrast_pool_size`、`--topo_grid_size`、`--topo_neighbor_k` 等（与
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
# 中的 *_convnextv2_nano_best.pth；OUTPUT 建议带实验名与日期便于对照。
"""
python ChangeDINO-main/scripts/infer_gf3_henan_tiles.py \
  --tiles-root datasets/GF3_Henan_CD_infer \
  --checkpoint ChangeDINO-main/checkpoints/S1GFloods-topo-b8-20260408/S1GFloods-topo-b8-20260408_convnextv2_nano_best.pth \
  --stats_file datasets/S1GFloods_CD_DINO/channel_stats_s1gfloods_train.json \
  --gpu_ids 0 \
  --batch_size 8 \
  --output-dir ChangeDINO-main/outputs/gf3_henan_contrast_b8

python ChangeDINO-main/scripts/infer_gf3_henan_tiles.py \
  --tiles-root datasets/GF3_Zhuozhou_CD_infer \
  --checkpoint ChangeDINO-main/checkpoints/S1GFloods-contrast-b8-YYYYMMDD/S1GFloods-contrast-b8-YYYYMMDD_convnextv2_nano_best.pth \
  --stats_file datasets/S1GFloods_CD_DINO/channel_stats_s1gfloods_train.json \
  --gpu_ids 0 \
  --batch_size 8 \
  --output-dir ChangeDINO-main/outputs/gf3_zhuozhou_contrast_b8

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
