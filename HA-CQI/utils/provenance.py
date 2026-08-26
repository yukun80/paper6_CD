"""HA-CQI 数据与配置溯源辅助函数。"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from data.runtime_snapshot import build_runtime_data_snapshot


def sha256_file(path: str | Path) -> str:
    """流式计算文件 SHA256，避免大文件一次性载入内存。"""
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset_provenance(
    data_root: str | Path,
    dataset: str,
    *,
    runtime_snapshot: dict[str, Any] | None = None,
    stats_mode: str | None = None,
    stats_source: str | None = None,
) -> dict[str, Any]:
    """从实际目录生成轻量溯源；历史 manifest/report 不参与运行准入。"""
    dataset_dir = (Path(data_root) / dataset).resolve()
    snapshot = runtime_snapshot or build_runtime_data_snapshot(dataset_dir)
    split_summaries: dict[str, Any] = {}
    counts: dict[str, int] = {}
    for split, raw_summary in snapshot["splits"].items():
        summary = dict(raw_summary)
        summary.pop("filenames", None)
        split_summaries[str(split)] = summary
        counts[str(split)] = int(summary["samples"])
    return {
        "dataset": str(dataset),
        "dataset_dir": str(dataset_dir),
        "membership_authority": "train_val_directories",
        "manifest_enforced": False,
        "historical_split_report": str(dataset_dir / "split_report.json"),
        "runtime_snapshot_id": str(snapshot["runtime_snapshot_id"]),
        "train_image_snapshot_id": str(snapshot["train_image_snapshot_id"]),
        "counts": counts,
        "splits": split_summaries,
        "stats_mode": stats_mode,
        "stats_source": stats_source,
    }


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
