#!/usr/bin/env python3
"""S1 河南整景 SAR 推理预处理包装脚本。"""

from __future__ import annotations

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from prepare_sar_scene_infer import main as prepare_main

"""
python ChangeDINO-main/scripts/prepare_s1_henan_infer.py \
--src-root datasets/S1_Henan \
--pre-image Zhengzhou_S1GRD_ASCENDING_VH_pre.tif \
--post-image Zhengzhou_S1GRD_ASCENDING_VH_Post.tif \
--out-root datasets/S1_Henan_CD_infer \
--tile-size 256 \
--stride 128 \
--overwrite
"""


def main() -> None:
    prepare_main(
        defaults={
            "src_root": "datasets/S1_Henan",
            "pre_image": "Zhengzhou_S1GRD_ASCENDING_VH_pre.tif",
            "post_image": "Zhengzhou_S1GRD_ASCENDING_VH_Post.tif",
            "out_root": "datasets/S1_Henan_CD_infer",
            "scene_tag": "s1_henan",
        },
        description="Prepare S1 Henan tiles for ChangeDINO inference",
    )


if __name__ == "__main__":
    main()
