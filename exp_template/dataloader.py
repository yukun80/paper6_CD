"""UrbanSARFloods CH12 数据加载模块。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class SampleRecord:
    """单样本路径信息。"""

    sample_id: str
    class_group: str
    sar_path: Path
    gt_path: Path


class UrbanFloodSegDataset(Dataset):
    """UrbanSARFloods 三分类语义分割数据集。"""

    def __init__(
        self,
        data_root: str,
        split_file: str,
        mean: Sequence[float],
        std: Sequence[float],
        ignore_index: int = 255,
        crop_size: Optional[int] = 256,
        random_crop: bool = False,
        random_hflip: bool = False,
        random_vflip: bool = False,
        seed: int = 42,
    ) -> None:
        self.data_root = Path(data_root).resolve()
        self.split_file = _resolve_split_file(self.data_root, split_file)
        self.samples = _parse_split_file(self.data_root, self.split_file)

        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.ignore_index = int(ignore_index)
        self.crop_size = int(crop_size) if crop_size is not None else None
        self.random_crop = bool(random_crop)
        self.random_hflip = bool(random_hflip)
        self.random_vflip = bool(random_vflip)

        self.rng = np.random.default_rng(seed)
        if self.mean.shape != self.std.shape:
            raise ValueError("mean/std 维度不一致")
        if self.mean.ndim != 1:
            raise ValueError("mean/std 需要是一维列表")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        sar = _read_sar(sample.sar_path)
        gt = _read_gt(sample.gt_path)

        if sar.shape[0] != len(self.mean):
            raise ValueError(
                f"通道数不匹配: sample={sample.sar_path}, got={sar.shape[0]}, expected={len(self.mean)}"
            )

        sar, gt = _sanitize_invalid_pixels(sar, gt, self.mean, self.ignore_index)
        gt = _sanitize_label_values(gt, self.ignore_index)

        if self.crop_size is not None:
            sar, gt = _crop_pair(sar, gt, self.crop_size, self.random_crop, self.rng)

        if self.random_hflip and self.rng.random() < 0.5:
            sar = sar[:, :, ::-1]
            gt = gt[:, ::-1]
        if self.random_vflip and self.rng.random() < 0.5:
            sar = sar[:, ::-1, :]
            gt = gt[::-1, :]

        sar = (sar - self.mean[:, None, None]) / np.clip(self.std[:, None, None], a_min=1e-6, a_max=None)

        image = torch.from_numpy(np.ascontiguousarray(sar)).float()
        label = torch.from_numpy(np.ascontiguousarray(gt)).long()
        meta = {
            "sample_id": sample.sample_id,
            "class_group": sample.class_group,
            "sar_path": str(sample.sar_path),
            "gt_path": str(sample.gt_path),
        }
        return image, label, meta


def build_dataloaders(data_cfg: Dict, train_cfg: Dict) -> Tuple[DataLoader, DataLoader]:
    """根据配置构建训练与验证 DataLoader。"""

    common = dict(
        data_root=data_cfg["root"],
        mean=data_cfg["mean"],
        std=data_cfg["std"],
        ignore_index=data_cfg.get("ignore_index", 255),
        crop_size=data_cfg.get("crop_size", 256),
        seed=train_cfg.get("seed", 42),
    )

    train_ds = UrbanFloodSegDataset(
        split_file=data_cfg.get("train_split", "Train_dataset.txt"),
        random_crop=bool(data_cfg.get("random_crop", True)),
        random_hflip=bool(data_cfg.get("random_hflip", True)),
        random_vflip=bool(data_cfg.get("random_vflip", False)),
        **common,
    )
    val_ds = UrbanFloodSegDataset(
        split_file=data_cfg.get("val_split", "Valid_dataset.txt"),
        random_crop=False,
        random_hflip=False,
        random_vflip=False,
        **common,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg.get("batch_size", 8)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=bool(train_cfg.get("pin_memory", True)),
        drop_last=bool(train_cfg.get("drop_last", False)),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg.get("val_batch_size", 1)),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=bool(train_cfg.get("pin_memory", True)),
        drop_last=False,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader


def build_eval_dataloader(data_cfg: Dict, train_cfg: Dict, split: str = "val") -> DataLoader:
    """构建评估/可视化 DataLoader。"""

    split_key = "val_split" if split == "val" else "train_split"
    ds = UrbanFloodSegDataset(
        data_root=data_cfg["root"],
        split_file=data_cfg.get(split_key, "Valid_dataset.txt"),
        mean=data_cfg["mean"],
        std=data_cfg["std"],
        ignore_index=data_cfg.get("ignore_index", 255),
        crop_size=data_cfg.get("crop_size", 256),
        random_crop=False,
        random_hflip=False,
        random_vflip=False,
        seed=train_cfg.get("seed", 42),
    )
    return DataLoader(
        ds,
        batch_size=int(train_cfg.get("val_batch_size", 1)),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=bool(train_cfg.get("pin_memory", True)),
        drop_last=False,
        collate_fn=collate_fn,
    )


def collate_fn(batch):
    """自定义 collate，保留元信息列表。"""

    images, labels, metas = zip(*batch)
    return torch.stack(images, dim=0), torch.stack(labels, dim=0), list(metas)


def _resolve_split_file(data_root: Path, split_file: str) -> Path:
    split_path = Path(split_file)
    if split_path.is_absolute():
        return split_path
    return (data_root / split_path).resolve()


def _normalize_rel_path(line: str) -> Path:
    rel = line.strip().replace("\\", "/")
    while rel.startswith("../"):
        rel = rel[3:]
    return Path(rel)


def _parse_split_file(data_root: Path, split_path: Path) -> List[SampleRecord]:
    if not split_path.exists():
        raise FileNotFoundError(f"split 文件不存在: {split_path}")

    samples: List[SampleRecord] = []
    for line_no, raw in enumerate(split_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue

        gt_rel = _normalize_rel_path(line)
        if len(gt_rel.parts) < 3:
            raise ValueError(f"{split_path}:{line_no} 非法路径: {raw}")
        if gt_rel.parts[1] != "GT":
            raise ValueError(f"{split_path}:{line_no} 期望包含 /GT/ : {raw}")
        if not gt_rel.name.endswith("_GT.tif"):
            raise ValueError(f"{split_path}:{line_no} 期望 *_GT.tif: {raw}")

        class_group = gt_rel.parts[0]
        sample_id = gt_rel.name[: -len("_GT.tif")]
        sar_rel = Path(class_group) / "SAR" / f"{sample_id}_SAR.tif"

        gt_path = (data_root / gt_rel).resolve()
        sar_path = (data_root / sar_rel).resolve()
        if not gt_path.exists() or not sar_path.exists():
            raise FileNotFoundError(
                f"样本文件不存在: gt={gt_path.exists()} sar={sar_path.exists()} | line={raw}"
            )

        samples.append(
            SampleRecord(
                sample_id=sample_id,
                class_group=class_group,
                sar_path=sar_path,
                gt_path=gt_path,
            )
        )
    return samples


def _read_sar(path: Path) -> np.ndarray:
    with rasterio.open(path) as ds:
        arr = ds.read().astype(np.float32)
    return arr


def _read_gt(path: Path) -> np.ndarray:
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype(np.int64)
    return arr


def _sanitize_invalid_pixels(
    sar: np.ndarray,
    gt: np.ndarray,
    mean: np.ndarray,
    ignore_index: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """将 SAR 的 NaN/Inf 填充为通道均值，并把标签位置改为 ignore。"""

    finite = np.isfinite(sar)
    valid_mask = np.all(finite, axis=0)

    for c in range(sar.shape[0]):
        c_mask = ~finite[c]
        if np.any(c_mask):
            sar[c][c_mask] = mean[c]

    gt = gt.copy()
    gt[~valid_mask] = ignore_index
    return sar, gt


def _sanitize_label_values(gt: np.ndarray, ignore_index: int) -> np.ndarray:
    """仅保留 0/1/2 三类，其余标签统一设为 ignore。"""

    out = gt.copy()
    valid = np.isin(out, [0, 1, 2])
    out[~valid] = ignore_index
    return out


def _crop_pair(
    sar: np.ndarray,
    gt: np.ndarray,
    crop_size: int,
    random_crop: bool,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = gt.shape
    if crop_size > h or crop_size > w:
        raise ValueError(f"crop_size={crop_size} 大于图像大小 {(h, w)}")

    if random_crop:
        top = int(rng.integers(0, h - crop_size + 1))
        left = int(rng.integers(0, w - crop_size + 1))
    else:
        top = (h - crop_size) // 2
        left = (w - crop_size) // 2

    sar = sar[:, top : top + crop_size, left : left + crop_size]
    gt = gt[top : top + crop_size, left : left + crop_size]
    return sar, gt
