"""监督参考实验配置加载。"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict


DEFAULT_REFERENCE_CONFIG: Dict[str, Any] = {
    "reference_data": {
        "root": "datasets/urban_sar_floods_test/urban_sar_floods_test_tiles_512_band5_band7",
        "split_dir": "datasets/urban_sar_floods_test/urban_sar_floods_test_tiles_512_band5_band7/splits",
        "train_split": "train.txt",
        "val_split": "val.txt",
        "ref_bank_split": "ref_bank.txt",
        "val_ratio": 0.2,
        "seed": 42,
        "ignore_index": 255,
        "max_train_samples": -1,
        "max_val_samples": -1,
        "max_ref_samples": -1,
    },
    "target_data": {
        "root": "datasets/GF3_Henan/GF3_Henan_tiles_512",
        "max_samples": -1,
    },
    "model": {
        "in_channels": 8,
        "embed_dim": 32,
        "hidden_dim": 48,
    },
    "prompts": {
        "patch_radius": 6,
        "positive_quantile": 0.90,
        "negative_quantile": 0.45,
        "min_component_area": 20,
        "min_point_distance": 16,
        "max_positive_candidates": 18,
        "max_negative_candidates": 18,
        "initial_positive_points": 6,
        "initial_negative_points": 6,
        "low_confidence_max_score": 0.18,
        "proposal_threshold": 0.55,
    },
    "segmenter": {
        "backend": "sam",
        "sam_model_type": "vit_b",
        "sam_checkpoint": "PPO-main/segmenter/checkpoint/sam_vit_b_01ec64.pth",
        "sam_module_root": "PPO-main/segmenter",
    },
    "optimizer": {
        "max_steps": 8,
        "min_positive_points": 2,
        "max_positive_points": 10,
        "min_negative_points": 2,
        "max_negative_points": 10,
    },
    "policy": {
        "hidden_dim": 128,
        "dropout": 0.1,
        "stop_threshold": 0.0,
    },
    "train": {
        "device": "cuda",
        "num_workers": 0,
        "batch_size": 4,
        "epochs": 8,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "work_dir": "sar_prompt_flood/work_dir/reference_supervised",
        "log_interval": 10,
    },
    "inference": {
        "use_policy": True,
        "save_visuals": True,
        "max_samples": -1,
        "output_dir": "sar_prompt_flood/outputs/reference_supervised",
        "save_geotiff": True,
    },
    "runtime": {
        "seed": 42,
    },
}


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def load_reference_config(path: str | None = None) -> Dict[str, Any]:
    """加载监督参考实验配置。"""
    cfg = deepcopy(DEFAULT_REFERENCE_CONFIG)
    if not path:
        cfg["data"] = deepcopy(cfg["reference_data"])
        return cfg
    cfg_path = Path(path)
    user_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg = _deep_update(cfg, user_cfg)
    if "reference_data" not in cfg and "data" in cfg:
        cfg["reference_data"] = deepcopy(cfg["data"])
    if "data" not in cfg:
        cfg["data"] = deepcopy(cfg["reference_data"])
    return cfg
