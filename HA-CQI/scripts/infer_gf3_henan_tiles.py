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


# 从仓库根目录执行；checkpoint v2 会提供数据统计路径与选择阈值。
# 当前 best_primary 的 selection threshold 为 0.76，因此默认不传 --threshold。
# 仅在固定阈值对照实验时追加 --threshold 0.40，它会覆盖 checkpoint 阈值。
"""
conda activate hacqi

python HA-CQI/scripts/infer_gf3_henan_tiles.py \
  --tiles-root datasets/GF3_Henan_CD_infer \
  --checkpoint HA-CQI/checkpoints/S1GFloods-HA-CQI-B2-OSCD-DINO5-8-11-CLEAN-s1-20260826/S1GFloods-HA-CQI-B2-OSCD-DINO5-8-11-CLEAN-s1-20260826_efficientnet_b2_best_primary.pth \
  --gpu_ids 0 \
  --batch_size 12 \
  --num_workers 8 \
  --amp \
  --amp_dtype bf16 \
  --skip-tiles \
  --output-dir HA-CQI/outputs/gf3_henan_20260826

python HA-CQI/scripts/infer_gf3_henan_tiles.py \
  --tiles-root datasets/GF3_Zhuozhou_CD_infer \
  --checkpoint HA-CQI/checkpoints/S1GFloods-HA-CQI-B2-OSCD-DINO5-8-11-CLEAN-s1-20260826/S1GFloods-HA-CQI-B2-OSCD-DINO5-8-11-CLEAN-s1-20260826_efficientnet_b2_best_primary.pth \
  --gpu_ids 0 \
  --batch_size 12 \
  --num_workers 8 \
  --amp \
  --amp_dtype bf16 \
  --output-dir HA-CQI/outputs/gf3_zhuozhou_20260826

python HA-CQI/scripts/infer_gf3_henan_tiles.py \
  --tiles-root datasets/LT1_Guangxi_CD_infer \
  --checkpoint HA-CQI/checkpoints/S1GFloods-HA-CQI-B2-OSCD-DINO5-8-11-CLEAN-s1-20260826/S1GFloods-HA-CQI-B2-OSCD-DINO5-8-11-CLEAN-s1-20260826_efficientnet_b2_best_primary.pth \
  --gpu_ids 0 \
  --batch_size 12 \
  --num_workers 8 \
  --amp \
  --amp_dtype bf16 \
  --skip-tiles \
  --output-dir HA-CQI/outputs/lt1_guangxi_20260826_1

python HA-CQI/scripts/infer_gf3_henan_tiles.py \
  --tiles-root datasets/USA_Brazos_River_CD_infer \
  --checkpoint HA-CQI/checkpoints/S1GFloods-HA-CQI-B2-OSCD-DINO5-8-11-CLEAN-s1-20260826/S1GFloods-HA-CQI-B2-OSCD-DINO5-8-11-CLEAN-s1-20260826_efficientnet_b2_best_primary.pth \
  --gpu_ids 0 \
  --batch_size 12 \
  --num_workers 8 \
  --amp \
  --amp_dtype bf16 \
  --skip-tiles \
  --output-dir HA-CQI/outputs/USA_Brazos_River_20260826_1

python HA-CQI/scripts/infer_gf3_henan_tiles.py \
  --tiles-root datasets/USA_San_Jacinto_CD_infer \
  --checkpoint HA-CQI/checkpoints/S1GFloods-HA-CQI-B2-OSCD-DINO5-8-11-CLEAN-s1-20260826/S1GFloods-HA-CQI-B2-OSCD-DINO5-8-11-CLEAN-s1-20260826_efficientnet_b2_best_primary.pth \
  --gpu_ids 0 \
  --batch_size 12 \
  --num_workers 8 \
  --amp \
  --amp_dtype bf16 \
  --skip-tiles \
  --output-dir HA-CQI/outputs/USA_San_Jacinto_20260826_1
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
