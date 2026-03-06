"""配置与日志相关工具。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import yaml

from .runtime import now_str


def load_config(config_file: str) -> Dict:
    """读取 YAML 配置文件。"""

    with open(config_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"配置文件格式错误: {config_file}")
    return cfg


def save_yaml(path: str | Path, data: Dict) -> None:
    """保存 YAML。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def save_json(path: str | Path, data: Dict) -> None:
    """保存 JSON。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def resolve_path(path: str, project_root: str) -> str:
    """将相对路径解析为项目绝对路径。"""

    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str((Path(project_root) / p).resolve())


def make_run_dirs(cfg: Dict, project_root: str, work_dir_override: str = "") -> Dict[str, Path]:
    """创建运行目录，并返回子路径映射。"""

    work_cfg = cfg.get("work_dir", {})
    root = work_dir_override or work_cfg.get("root", "exp_template/work_dir")
    root_path = Path(resolve_path(root, project_root))

    exp_subdir = work_cfg.get("exp_subdir", "urban_sar_floods_ch12")
    model_name = cfg.get("model", {}).get("name", "model")
    backbone = cfg.get("model", {}).get("backbone", "default")
    run_template = work_cfg.get("run_name_template", "{model}_{backbone}_{time}")
    run_name = run_template.format(model=model_name, backbone=backbone, time=now_str())

    run_root = root_path / exp_subdir / run_name
    paths = {
        "root": run_root,
        "checkpoints": run_root / "checkpoints",
        "eval": run_root / "eval",
        "vis": run_root / "vis",
        "logs": run_root / "logs",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def log_line(log_file: str | Path, msg: str, also_print: bool = True) -> None:
    """写日志到文件并可选打印到终端。"""

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    lf = Path(log_file)
    lf.parent.mkdir(parents=True, exist_ok=True)
    with open(lf, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    if also_print:
        print(line)
