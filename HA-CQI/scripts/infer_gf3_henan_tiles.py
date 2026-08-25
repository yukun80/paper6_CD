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


# 示例：checkpoint v2 会提供数据统计路径与选择阈值。
"""
python HA-CQI/scripts/infer_gf3_henan_tiles.py \
  --tiles-root datasets/GF3_Henan_CD_infer \
  --checkpoint HA-CQI/checkpoints/<b2_run>/<b2_run>_efficientnet_b2_best_primary.pth \
  --gpu_ids 0 \
  --batch_size 8 \
  --output-dir HA-CQI/outputs/gf3_henan_corrected

python HA-CQI/scripts/infer_gf3_henan_tiles.py \
  --tiles-root datasets/GF3_Zhuozhou_CD_infer \
  --checkpoint HA-CQI/checkpoints/<b2_run>/<b2_run>_efficientnet_b2_best_primary.pth \
  --gpu_ids 0 \
  --batch_size 8 \
  --output-dir HA-CQI/outputs/gf3_zhuozhou_corrected_0823
"""


def main() -> None:
    infer_main(
        defaults={
            "tiles_root": "datasets/GF3_Henan_CD_infer",
            "output_dir": PROJECT_ROOT / "outputs" / "gf3_henan",
        },
        description="Infer HA-CQI on GF3 Henan SAR tiles",
    )


if __name__ == "__main__":
    main()
