#!/usr/bin/env python3
"""GF3 河南整景 SAR 变化检测推理包装脚本。"""

from __future__ import annotations

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from infer_sar_scene_tiles import main as infer_main


# 示例：将 CHECKPOINT 换为 HA-CQI 训练目录中的实际 best 权重。
"""
python HA-CQI/scripts/infer_gf3_henan_tiles.py \
  --tiles-root datasets/GF3_Henan_CD_infer \
  --checkpoint HA-CQI/checkpoints/S1GFloods-HA-CQI-vits16-20260427/S1GFloods-HA-CQI-vits16-20260427_efficientnet_b0_best_iou.pth \
  --stats_file datasets/S1GFloods_CD_DINO/channel_stats_s1gfloods_train.json \
  --gpu_ids 0 \
  --batch_size 6 \
  --threshold 0.40 \
  --output-dir HA-CQI/outputs/gf3_henan_ha_cqi_0427_best_iou
  
python HA-CQI/scripts/infer_gf3_henan_tiles.py \
  --tiles-root datasets/GF3_Zhuozhou_CD_infer \
  --checkpoint HA-CQI/checkpoints/S1GFloods-HA-CQI-vits16-20260427/S1GFloods-HA-CQI-vits16-20260427_efficientnet_b0_epoch80.pth \
  --stats_file datasets/S1GFloods_CD_DINO/channel_stats_s1gfloods_train.json \
  --gpu_ids 0 \
  --batch_size 6 \
  --threshold 0.40 \
  --output-dir HA-CQI/outputs/gf3_zhuozhou_ha_cqi_0427_epoch80
"""


def main() -> None:
    infer_main(
        defaults={
            "tiles_root": "datasets/GF3_Henan_CD_infer",
            "output_dir": PROJECT_ROOT / "outputs" / "gf3_henan",
            "threshold": 0.40,
            "eval_fg_threshold": 0.40,
        },
        description="Infer HA-CQI on GF3 Henan SAR tiles",
    )


if __name__ == "__main__":
    main()
