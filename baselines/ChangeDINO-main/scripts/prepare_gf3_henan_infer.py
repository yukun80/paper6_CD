#!/usr/bin/env python3
"""GF3 河南整景 SAR 推理预处理包装脚本。"""

from __future__ import annotations

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from prepare_sar_scene_infer import main as prepare_main

"""
python ChangeDINO-main/scripts/prepare_gf3_henan_infer.py \
--src-root datasets/GF3_Henan \
--pre-image Pre_ZhengzhouC_descending.tif \
--post-image Post_ZhengzhouC_descending.tif \
--out-root datasets/GF3_Henan_CD_infer \
--tile-size 256 \
--stride 128 \
--overwrite

python ChangeDINO-main/scripts/prepare_gf3_henan_infer.py \
--src-root datasets/GF3_Henan \
--pre-image Pre_Zhengzhou_ascending_s1_radmatch.tif \
--post-image Post_Zhengzhou_descending_clip.tif \
--out-root datasets/GF3_Henan_CD_infer_pre_value_only \
--tile-size 256 \
--stride 128 \
--pre-stretch-mode value \
--pre-value-min 0.2 \
--pre-value-max 2.0 \
--overwrite

python ChangeDINO-main/scripts/prepare_gf3_henan_infer.py \
--src-root datasets/GF3_Zhuozhou \
--pre-image Pre_Zhuozhou_clip.tif \
--post-image Post_Zhuozhou_clip.tif \
--out-root datasets/GF3_Zhuozhou_CD_infer \
--tile-size 256 \
--stride 128 \
--overwrite


"""


def main() -> None:
    prepare_main(
        defaults={
            "src_root": "datasets/GF3_Henan",
            "pre_image": "Pre_Zhengzhou_ascending_s1_radmatch.tif",
            "post_image": "Post_Zhengzhou_descending_clip.tif",
            "out_root": "datasets/GF3_Henan_CD_infer",
            "scene_tag": "gf3_henan",
        },
        description="Prepare GF3 Henan tiles for ChangeDINO inference",
    )


if __name__ == "__main__":
    main()
