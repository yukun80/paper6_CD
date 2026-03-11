"""GF3_Henan-only 配置加载。"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict


DEFAULT_CONFIG: Dict[str, Any] = {
    "data": {
        "pair_root": "datasets/GF3_Henan",
        "pre_name": "Pre_Zhengzhou_ascending_clip.tif",
        "post_name": "Post_Zhengzhou_ascending_clip.tif",
        "processed_root": "datasets/GF3_Henan_processed",
    },
    "preprocess": {
        "tile_size": 512,
        "overlap": 128,
        "min_valid_ratio": 0.6,
        "clip_percentiles": [1.0, 99.0],
        "max_tiles": -1,
    },
    "prompts": {
        "patch_radius": 6,
        "positive_quantile": 0.92,
        "negative_quantile": 0.35,
        "min_component_area": 24,
        "min_point_distance": 16,
        "max_positive_candidates": 18,
        "max_negative_candidates": 18,
        "initial_positive_points": 6,
        "initial_negative_points": 6,
        "low_confidence_max_score": 0.18,
    },
    "segmenter": {
        "backend": "sam",
        "sam_model_type": "vit_b",
        "sam_checkpoint": "",
        "sam_module_root": "PPO-main/segmenter",
    },
    "optimizer": {
        "max_steps": 10,
        "min_positive_points": 2,
        "max_positive_points": 10,
        "min_negative_points": 2,
        "max_negative_points": 10,
    },
    "inference": {
        "mosaic_threshold": 0.5,
        "max_tiles": -1,
        "save_tile_visuals": True,
    },
    "runtime": {
        "seed": 42,
        "num_workers": 0,
        "output_dir": "sar_prompt_flood/outputs/gf3_henan",
    },
}


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def load_config(path: str | None = None) -> Dict[str, Any]:
    """加载 JSON 配置，并与 GF3 默认配置合并。"""
    cfg = deepcopy(DEFAULT_CONFIG)
    if not path:
        return cfg
    cfg_path = Path(path)
    user_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    return _deep_update(cfg, user_cfg)
