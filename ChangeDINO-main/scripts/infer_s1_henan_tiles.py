#!/usr/bin/env python3
"""S1 河南整景 SAR 变化检测推理包装脚本。"""

from __future__ import annotations

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from infer_sar_scene_tiles import main as infer_main

"""
python ChangeDINO-main/scripts/infer_s1_henan_tiles.py \
--tiles-root datasets/S1_Henan_CD_infer \
--checkpoint ChangeDINO-main/checkpoints/S1GFloods-ChangeDINO/S1GFloods-ChangeDINO_mobilenetv2_best.pth \
--stats_file datasets/S1GFloods_CD_DINO/channel_stats_s1gfloods_train.json \
--gpu_ids 0 \
--batch_size 8 \
--output-dir ChangeDINO-main/outputs/s1_henan
"""


def main() -> None:
    infer_main(
        defaults={
            "tiles_root": "datasets/S1_Henan_CD_infer",
            "output_dir": "ChangeDINO-main/outputs/s1_henan",
        },
        description="Infer ChangeDINO on S1 Henan SAR tiles",
    )


if __name__ == "__main__":
    main()
