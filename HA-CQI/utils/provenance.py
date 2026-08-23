"""HA-CQI 数据与配置溯源辅助函数。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    """流式计算文件 SHA256，避免大文件一次性载入内存。"""
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {source}")
    return payload


def load_dataset_provenance(data_root: str | Path, dataset: str) -> dict[str, Any]:
    """读取数据构建报告，并为旧数据集提供可诊断的兼容结果。"""
    dataset_dir = Path(data_root) / dataset
    report_path = dataset_dir / "split_report.json"
    result: dict[str, Any] = {
        "dataset": str(dataset),
        "dataset_dir": str(dataset_dir),
        "split_report": str(report_path),
        "dataset_fingerprint": None,
        "counts": {},
        "counts_by_label_presence": {},
    }
    if not report_path.is_file():
        return result

    report = load_json(report_path)
    result.update(
        {
            "dataset_fingerprint": report.get("dataset_fingerprint"),
            "counts": report.get("counts", {}),
            "counts_by_label_presence": report.get("counts_by_label_presence", {}),
            "split_report_sha256": sha256_file(report_path),
        }
    )
    for split in ("train", "val"):
        manifest_path = dataset_dir / f"manifest_{split}.csv"
        if manifest_path.is_file():
            result[f"manifest_{split}_sha256"] = sha256_file(manifest_path)
    return result


def portable_repo_path(path_like: str | Path | None, repo_root: str | Path) -> str | None:
    """仓库内路径写成相对路径，仓库外路径保持绝对形式。"""
    if path_like in (None, ""):
        return None
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        return str(path)
    root = Path(repo_root).resolve()
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path)
