import copy
import os
from pathlib import Path
from typing import Any, Dict

import yaml


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并字典，后者覆盖前者。"""
    out = copy.deepcopy(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_config(config_file: str) -> Dict[str, Any]:
    """加载配置，支持 `base_config` 继承。"""
    cfg = load_yaml(config_file)
    base_cfg_file = cfg.get("base_config")
    if base_cfg_file:
        base_path = Path(config_file).resolve().parent / base_cfg_file
        base_cfg = load_config(str(base_path))
        cfg = _deep_update(base_cfg, {k: v for k, v in cfg.items() if k != "base_config"})
    return cfg


def resolve_path(path: str, project_root: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(project_root, path))


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def dump_config(cfg: Dict[str, Any], output_file: str) -> None:
    with open(output_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
