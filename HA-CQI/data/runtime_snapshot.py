"""按实际目录成员解析训练数据，并提供非阻塞统计快照。"""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
import rasterio

from .tif_io import (
    build_valid_mask,
    read_binary_label_tif_with_valid_mask,
    stretch_sar_array,
)


VALID_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class SplitFiles:
    """一个 split 中严格同名的 A/B/label 三元组。"""

    split: str
    a: dict[str, Path]
    b: dict[str, Path]
    label: dict[str, Path]
    filenames: tuple[str, ...]


def _scan_image_dir(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Split directory not found: {directory}")
    files = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_SUFFIXES
    )
    return {path.name: path for path in files}


def _resolve_label_dir(split_root: Path) -> Path:
    for candidate in (split_root / "label", split_root / "Label"):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Cannot find label directory under: {split_root}")


def scan_split_files(dataset_root: str | Path, split: str) -> SplitFiles:
    """以目录为成员权威，同时拒绝不完整的时相/标签三元组。"""
    split_root = Path(dataset_root) / split
    a_map = _scan_image_dir(split_root / "A")
    b_map = _scan_image_dir(split_root / "B")
    label_map = _scan_image_dir(_resolve_label_dir(split_root))
    a_names = set(a_map)
    b_names = set(b_map)
    label_names = set(label_map)
    if a_names != b_names or a_names != label_names:
        missing_in_a = sorted((b_names | label_names) - a_names)
        missing_in_b = sorted((a_names | label_names) - b_names)
        missing_in_label = sorted((a_names | b_names) - label_names)
        raise ValueError(
            f"File mismatch under split {split_root}: A/B/label must have identical "
            "file names. "
            f"missing_in_A={missing_in_a[:10]}, missing_in_B={missing_in_b[:10]}, "
            f"missing_in_label={missing_in_label[:10]}"
        )
    filenames = tuple(sorted(a_names))
    if not filenames:
        raise ValueError(f"No samples found under split: {split_root}")
    return SplitFiles(split, a_map, b_map, label_map, filenames)


def _update_file_digest(
    digest: "hashlib._Hash",
    dataset_root: Path,
    split: str,
    role: str,
    paths: Iterable[Path],
) -> None:
    for path in paths:
        stat = path.stat()
        relative = path.relative_to(dataset_root).as_posix()
        record = (
            f"{split}\0{role}\0{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n"
        )
        digest.update(record.encode("utf-8"))


def _label_summary(paths: Iterable[Path]) -> dict[str, int | float]:
    sample_count = 0
    background_tiles = 0
    foreground_pixels = 0
    valid_pixels = 0
    for path in paths:
        if path.suffix.lower() in {".tif", ".tiff"}:
            label, valid = read_binary_label_tif_with_valid_mask(path)
        else:
            raw = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
            valid = np.ones(raw.shape, dtype=bool)
            label = (raw > 0).astype(np.uint8)
        current_foreground = int(np.count_nonzero((label == 1) & valid))
        sample_count += 1
        background_tiles += int(current_foreground == 0)
        foreground_pixels += current_foreground
        valid_pixels += int(np.count_nonzero(valid))
    return {
        "samples": sample_count,
        "background_tiles": background_tiles,
        "foreground_tiles": sample_count - background_tiles,
        "foreground_pixels": foreground_pixels,
        "valid_pixels": valid_pixels,
        "foreground_ratio": (
            float(foreground_pixels / valid_pixels) if valid_pixels else 0.0
        ),
    }


def build_runtime_data_snapshot(dataset_root: str | Path) -> dict[str, object]:
    """记录本次实际使用成员；snapshot 只用于缓存和审计，不参与准入。"""
    root = Path(dataset_root).resolve()
    runtime_digest = hashlib.sha256()
    train_image_digest = hashlib.sha256()
    split_payloads: dict[str, object] = {}
    for split in ("train", "val"):
        files = scan_split_files(root, split)
        for role, mapping in (("A", files.a), ("B", files.b), ("label", files.label)):
            ordered = [mapping[name] for name in files.filenames]
            _update_file_digest(runtime_digest, root, split, role, ordered)
            if split == "train" and role in {"A", "B"}:
                _update_file_digest(train_image_digest, root, split, role, ordered)
        split_payloads[split] = {
            **_label_summary(files.label[name] for name in files.filenames),
            "filenames": list(files.filenames),
        }
    return {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root),
        "membership_authority": "train_val_directories",
        "manifest_enforced": False,
        "runtime_snapshot_id": runtime_digest.hexdigest(),
        "train_image_snapshot_id": train_image_digest.hexdigest(),
        "splits": split_payloads,
    }


def _load_image_rgb(path: Path) -> np.ndarray:
    if path.suffix.lower() not in {".tif", ".tiff"}:
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
        return image.reshape(-1, 3)
    with rasterio.open(path) as dataset:
        array = dataset.read(1).astype(np.float32, copy=False)
        valid = build_valid_mask(array, dataset.nodata)
    stretched = stretch_sar_array(array, valid)
    rgb = np.repeat(stretched[:, :, None], 3, axis=2).astype(
        np.float64,
        copy=False,
    )
    return rgb.reshape(-1, 3)


def compute_stats_for_paths(
    dataset_root: str | Path,
    split: str,
    paths: Iterable[Path],
    *,
    num_pairs: int | None = None,
) -> dict[str, object]:
    """对显式图像集合计算输入 mean/std。"""
    root = Path(dataset_root).resolve()
    selected_paths = list(paths)
    if not selected_paths:
        raise ValueError(f"No images found for split={split}: {root}")
    sums = np.zeros(3, dtype=np.float64)
    squared_sums = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    for path in selected_paths:
        values = _load_image_rgb(path)
        sums += values.sum(axis=0)
        squared_sums += np.square(values).sum(axis=0)
        pixel_count += values.shape[0]
    mean = sums / pixel_count
    variance = np.maximum(squared_sums / pixel_count - np.square(mean), 0.0)
    std = np.maximum(np.sqrt(variance), 1e-6)
    return {
        "format_version": 3,
        "data_root": str(root),
        "split": split,
        "num_pairs": num_pairs,
        "num_images": len(selected_paths),
        "pixel_count": int(pixel_count),
        "recommended_config_fields": {
            "mean": [float(value) for value in mean],
            "std": [float(value) for value in std],
        },
    }


def compute_channel_stats(
    dataset_root: str | Path,
    *,
    split_files: SplitFiles | None = None,
) -> dict[str, object]:
    """按当前 train 三元组中的 A/B 计算模型输入 mean/std。"""
    root = Path(dataset_root).resolve()
    files = split_files or scan_split_files(root, "train")
    paths = [files.a[name] for name in files.filenames] + [
        files.b[name] for name in files.filenames
    ]
    return compute_stats_for_paths(
        root,
        "train",
        paths,
        num_pairs=len(files.filenames),
    )


def atomic_write_json(payload: dict[str, object], output: str | Path) -> Path:
    """原子写 JSON，避免统计计算中断留下半文件。"""
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return target


def resolve_auto_channel_stats(
    dataset_root: str | Path,
    cache_dir: str | Path,
) -> tuple[dict[str, object], dict[str, object], Path, bool]:
    """按当前目录快照命中或生成 stats；snapshot ID 不用于拒绝训练。"""
    snapshot = build_runtime_data_snapshot(dataset_root)
    cache_path = Path(cache_dir) / (
        f"channel_stats_{snapshot['train_image_snapshot_id']}.json"
    )
    cache_hit = False
    if cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = {}
        recommended = cached.get("recommended_config_fields", {})
        cached_mean = recommended.get("mean")
        cached_std = recommended.get("std")
        try:
            values_valid = (
                isinstance(cached_mean, list)
                and isinstance(cached_std, list)
                and len(cached_mean) == 3
                and len(cached_std) == 3
                and bool(
                    np.isfinite(
                        np.asarray(cached_mean + cached_std, dtype=np.float64)
                    ).all()
                )
                and all(float(value) > 0.0 for value in cached_std)
            )
        except (TypeError, ValueError):
            values_valid = False
        if (
            cached.get("train_image_snapshot_id") == snapshot["train_image_snapshot_id"]
            and values_valid
        ):
            stats = cached
            cache_hit = True
        else:
            stats = {}
    else:
        stats = {}
    if not stats:
        stats = compute_channel_stats(dataset_root)
        stats["runtime_snapshot_id"] = snapshot["runtime_snapshot_id"]
        stats["train_image_snapshot_id"] = snapshot["train_image_snapshot_id"]
        atomic_write_json(stats, cache_path)
    return stats, snapshot, cache_path, cache_hit


def warn_if_runtime_snapshot_changed(
    checkpoint_snapshot_id: str | None,
    current_snapshot_id: str,
) -> bool:
    """数据成员变化只警告，不把动态目录变成 resume 准入条件。"""
    if checkpoint_snapshot_id == current_snapshot_id:
        return False
    warnings.warn(
        "Resume data members differ from the checkpoint snapshot; training will "
        "continue with the current A/B/label directories and is no longer a "
        "strictly reproducible continuation. "
        f"checkpoint={checkpoint_snapshot_id!r}, current={current_snapshot_id!r}",
        UserWarning,
        stacklevel=2,
    )
    return True
