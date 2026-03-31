#!/usr/bin/env python3
"""GF3 河南整景 SAR 变化检测推理包装脚本。"""

from __future__ import annotations

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from infer_sar_scene_tiles import main as infer_main

"""
python baselines/ChangeDINO_raw/scripts/infer_gf3_henan_tiles.py \
--tiles-root datasets/GF3_Henan_CD_infer \
--checkpoint baselines/ChangeDINO_raw/checkpoints/S1GFloods-ChangeDINO-vits16/S1GFloods-ChangeDINO-vits16_mobilenetv2_epoch_010.pth \
--stats_file datasets/S1GFloods_CD_DINO/channel_stats_s1gfloods_train.json \
--gpu_ids 0 \
--batch_size 8 \
--output-dir baselines/ChangeDINO_raw/outputs/gf3_henan_vits16

python baselines/ChangeDINO_raw/scripts/infer_gf3_henan_tiles.py \
--tiles-root datasets/GF3_Zhuozhou_CD_infer \
--checkpoint baselines/ChangeDINO_raw/checkpoints/S1GFloods-ChangeDINO-vits16/S1GFloods-ChangeDINO-vits16_mobilenetv2_epoch_010.pth \
--stats_file datasets/S1GFloods_CD_DINO/channel_stats_s1gfloods_train.json \
--gpu_ids 0 \
--batch_size 8 \
--output-dir baselines/ChangeDINO_raw/outputs/gf3_zhuozhou_vits16
"""


def main() -> None:
    infer_main(
        defaults={
            "tiles_root": "datasets/GF3_Henan_CD_infer",
            "output_dir": "baselines/ChangeDINO_raw/outputs/gf3_henan",
        },
        description="Infer ChangeDINO on GF3 Henan SAR tiles",
    )


if __name__ == "__main__":
    main()
