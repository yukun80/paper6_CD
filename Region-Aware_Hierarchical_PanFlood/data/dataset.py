import os
from glob import glob
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from data.channel_specs import build_channel_id_arrays, build_channel_layout


def _load_split_entries(data_root: str, split_file: str) -> List[str]:
    # 优先按传入路径解析，存在则直接使用；否则相对 data_root 解析。
    if os.path.exists(split_file):
        split_path = split_file
    else:
        split_path = split_file if os.path.isabs(split_file) else os.path.join(data_root, split_file)
    with open(split_path, "r", encoding="utf-8") as f:
        return [x.strip() for x in f if x.strip()]


def _resolve_pairs(data_root: str, split_file: str, gt_dir: str = "GT", sar_dir: str = "SAR") -> List[Tuple[str, str]]:
    """按文件名索引 split，避免 '../xx/GT/*.tif' 相对路径歧义。"""
    entries = _load_split_entries(data_root, split_file)
    gt_files = glob(os.path.join(data_root, "*", gt_dir, "*_GT.tif"))
    gt_map = {os.path.basename(p): p for p in gt_files}

    pairs = []
    for line in entries:
        gt_name = os.path.basename(line)
        if gt_name not in gt_map:
            raise FileNotFoundError(f"Cannot resolve GT from split line: {line}")
        gt_path = gt_map[gt_name]
        sar_path = gt_path.replace(f"{os.sep}{gt_dir}{os.sep}", f"{os.sep}{sar_dir}{os.sep}").replace("_GT.tif", "_SAR.tif")
        if not os.path.exists(sar_path):
            raise FileNotFoundError(f"SAR not found for GT: {gt_path}")
        pairs.append((sar_path, gt_path))
    return pairs


def _safe_norm(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    std = torch.clamp(std, min=1e-6)
    return (x - mean) / std


def _compute_engineered_from_8ch(imgs: torch.Tensor) -> torch.Tensor:
    """基于8通道构造4个工程通道。

    通道顺序（实测确认）：
    0 pre_coh_vh, 1 pre_coh_vv, 2 co_coh_vh, 3 co_coh_vv,
    4 pre_int_vh, 5 pre_int_vv, 6 co_int_vh, 7 co_int_vv
    """
    pre_coh_vh, pre_coh_vv, co_coh_vh, co_coh_vv = imgs[0], imgs[1], imgs[2], imgs[3]
    pre_int_vh, pre_int_vv, co_int_vh, co_int_vv = imgs[4], imgs[5], imgs[6], imgs[7]

    # 数据中无显式 post，这里采用 co-event 作为 post 近似状态。
    dI_vv = co_int_vv - pre_int_vv
    dI_vh = co_int_vh - pre_int_vh
    dC_vv = pre_coh_vv - co_coh_vv
    dC_vh = pre_coh_vh - co_coh_vh

    return torch.stack([dI_vv, dI_vh, dC_vv, dC_vh], dim=0)


class UrbanSARFloodsDataset(Dataset):
    """UrbanSARFloods 数据集读取器（支持 8ch / 8+4 / 12ch）。"""

    def __init__(
        self,
        data_root: str,
        split_file: str,
        input_mode: str,
        mean: Sequence[float],
        std: Sequence[float],
        crop_size: int = 252,
        random_crop: bool = False,
        random_hflip: bool = False,
        random_vflip: bool = False,
        ignore_index: int = 255,
        auto_label_mapping: bool = False,
        seed: int = 42,
    ) -> None:
        self.data_root = data_root
        self.pairs = _resolve_pairs(data_root, split_file)
        self.input_mode = input_mode
        self.layout = build_channel_layout(input_mode)
        self.role_ids = build_channel_id_arrays(self.layout)

        self.mean = torch.tensor(list(mean), dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(list(std), dtype=torch.float32).view(-1, 1, 1)
        if len(self.mean) != len(self.layout):
            raise ValueError(f"mean/std length ({len(self.mean)}) mismatch channels ({len(self.layout)})")

        self.crop_size = int(crop_size)
        self.random_crop = bool(random_crop)
        self.random_hflip = bool(random_hflip)
        self.random_vflip = bool(random_vflip)
        self.ignore_index = int(ignore_index)
        self.auto_label_mapping = bool(auto_label_mapping)
        self.rng = np.random.default_rng(seed)

        self.label_mapping: Dict[int, int] = self._infer_label_mapping() if self.auto_label_mapping else {
            0: 0,
            1: 1,
            2: 2,
        }
        self._identity_mapping = self.label_mapping == {0: 0, 1: 1, 2: 2}

        # 预计算样本级统计，供采样器使用。
        self.sample_stats = self._build_sample_stats()

    def __len__(self) -> int:
        return len(self.pairs)

    def _build_sample_stats(self) -> List[Dict[str, float]]:
        stats = []
        for _, gt_path in self.pairs:
            with rasterio.open(gt_path) as ds:
                gt = ds.read(1).astype(np.int64)
            gt = self._apply_label_mapping_numpy(gt)
            total = float(gt.size)
            num_open = float((gt == 1).sum())
            num_urban = float((gt == 2).sum())
            stats.append(
                {
                    "open_ratio": num_open / max(total, 1.0),
                    "urban_ratio": num_urban / max(total, 1.0),
                    "has_urban": float(num_urban > 0),
                }
            )
        return stats

    def compute_class_histogram(self, num_classes: int = 3) -> torch.Tensor:
        hist = torch.zeros(num_classes, dtype=torch.float64)
        for _, gt_path in self.pairs:
            with rasterio.open(gt_path) as ds:
                gt = ds.read(1).astype(np.int64)
            gt = self._apply_label_mapping_numpy(gt)
            for c in range(num_classes):
                hist[c] += float((gt == c).sum())
        return hist.float()

    def _infer_label_mapping(self) -> Dict[int, int]:
        """自动推断标签映射，兼容非标准编码（如二类或其他离散值）。"""
        uniq = set()
        for _, gt_path in self.pairs:
            with rasterio.open(gt_path) as ds:
                gt = ds.read(1)
            vals = np.unique(gt).tolist()
            uniq.update(int(v) for v in vals)
        uniq.discard(self.ignore_index)
        uniq_sorted = sorted(uniq)

        if uniq_sorted == [0, 1, 2]:
            return {0: 0, 1: 1, 2: 2}
        if len(uniq_sorted) == 2:
            # 二分类口径：背景->0，洪水->1（urban 类缺失）。
            return {uniq_sorted[0]: 0, uniq_sorted[1]: 1}
        if len(uniq_sorted) == 3:
            return {uniq_sorted[0]: 0, uniq_sorted[1]: 1, uniq_sorted[2]: 2}

        # 超出3类时，按排序前3个映射，其余落入 ignore。
        mapping = {}
        for i, raw in enumerate(uniq_sorted[:3]):
            mapping[raw] = i
        return mapping

    def get_label_mapping_report(self) -> Dict:
        return {
            "auto_label_mapping": self.auto_label_mapping,
            "ignore_index": self.ignore_index,
            "mapping": {int(k): int(v) for k, v in self.label_mapping.items()},
        }

    def _apply_label_mapping(self, gt: torch.Tensor) -> torch.Tensor:
        if self._identity_mapping and self.ignore_index == 255:
            return gt
        mapped = torch.full_like(gt, fill_value=self.ignore_index)
        for raw, new in self.label_mapping.items():
            mapped[gt == int(raw)] = int(new)
        mapped[gt == self.ignore_index] = self.ignore_index
        return mapped

    def _apply_label_mapping_numpy(self, gt: np.ndarray) -> np.ndarray:
        if self._identity_mapping and self.ignore_index == 255:
            return gt
        mapped = np.full_like(gt, fill_value=self.ignore_index)
        for raw, new in self.label_mapping.items():
            mapped[gt == int(raw)] = int(new)
        mapped[gt == self.ignore_index] = self.ignore_index
        return mapped

    def _read_item(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
        sar_path, gt_path = self.pairs[index]
        with rasterio.open(sar_path) as ds:
            imgs = ds.read().astype(np.float32)
        with rasterio.open(gt_path) as ds:
            gt = ds.read(1).astype(np.int64)

        imgs = torch.from_numpy(imgs)
        gt = torch.from_numpy(gt)
        gt = self._apply_label_mapping(gt)

        if self.input_mode == "8ch":
            if imgs.shape[0] != 8:
                raise ValueError(f"Expect 8 channels for mode=8ch, got {imgs.shape[0]} @ {sar_path}")
        elif self.input_mode == "8ch_plus_engineered":
            if imgs.shape[0] != 8:
                raise ValueError(f"Expect base 8 channels for engineered mode, got {imgs.shape[0]} @ {sar_path}")
            eng = _compute_engineered_from_8ch(imgs)
            imgs = torch.cat([imgs, eng], dim=0)
        elif self.input_mode == "12ch":
            if imgs.shape[0] != 12:
                raise ValueError(f"Expect 12 channels for mode=12ch, got {imgs.shape[0]} @ {sar_path}")
        else:
            raise ValueError(f"Unsupported mode: {self.input_mode}")

        return imgs, gt, sar_path, gt_path

    def _crop(self, imgs: torch.Tensor, gt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h, w = gt.shape[-2:]
        ch = cw = self.crop_size
        if ch > h or cw > w:
            raise ValueError(f"Crop size {self.crop_size} exceeds image size {(h, w)}")

        if self.random_crop:
            top = int(self.rng.integers(0, h - ch + 1))
            left = int(self.rng.integers(0, w - cw + 1))
        else:
            top = (h - ch) // 2
            left = (w - cw) // 2

        imgs = imgs[:, top : top + ch, left : left + cw]
        gt = gt[top : top + ch, left : left + cw]
        return imgs, gt

    def _flip(self, imgs: torch.Tensor, gt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.random_hflip and self.rng.random() < 0.5:
            imgs = torch.flip(imgs, dims=(-1,))
            gt = torch.flip(gt, dims=(-1,))
        if self.random_vflip and self.rng.random() < 0.5:
            imgs = torch.flip(imgs, dims=(-2,))
            gt = torch.flip(gt, dims=(-2,))
        return imgs, gt

    def __getitem__(self, index: int):
        imgs, gt, sar_path, gt_path = self._read_item(index)

        finite = torch.isfinite(imgs)
        valid_mask = finite.all(dim=0)

        fill_vals = self.mean.expand_as(imgs)
        imgs = torch.where(finite, imgs, fill_vals)

        gt = gt.clone()
        gt[~valid_mask] = self.ignore_index

        imgs, gt = self._crop(imgs, gt)
        imgs, gt = self._flip(imgs, gt)

        imgs = _safe_norm(imgs, self.mean, self.std)

        c = imgs.shape[0]
        x_dict = {
            "imgs": imgs,
            "chn_ids": torch.tensor(self.role_ids["chn_ids"][:c], dtype=torch.long),
            "time_ids": torch.tensor(self.role_ids["time_ids"][:c], dtype=torch.long),
            "feature_type_ids": torch.tensor(self.role_ids["feature_type_ids"][:c], dtype=torch.long),
            "temporal_role_ids": torch.tensor(self.role_ids["temporal_role_ids"][:c], dtype=torch.long),
            "polarization_ids": torch.tensor(self.role_ids["polarization_ids"][:c], dtype=torch.long),
            "source_role_ids": torch.tensor(self.role_ids["source_role_ids"][:c], dtype=torch.long),
        }

        meta = {
            "sar_path": sar_path,
            "gt_path": gt_path,
            "channel_names": self.role_ids["channel_names"][:c],
            "invalid_ratio": float((gt == self.ignore_index).float().mean().item()),
        }

        return x_dict, gt.long(), meta


def collate_fn(batch):
    """将样本打包为训练输入。"""
    x_list, y_list, m_list = zip(*batch)
    keys = x_list[0].keys()
    x_out = {}
    for k in keys:
        x_out[k] = torch.stack([x[k] for x in x_list], dim=0)
    y_out = torch.stack(y_list, dim=0)
    return x_out, y_out, list(m_list)
