import os
from glob import glob
from typing import Dict, List, Sequence, Tuple

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


def _resolve_paths(data_root: str, split_file: str) -> List[Tuple[str, str]]:
    """解析 split 文件，返回 (SAR, GT) 成对路径列表。"""
    split_path = split_file if os.path.isabs(split_file) else os.path.join(data_root, split_file)
    with open(split_path, "r", encoding="utf-8") as f:
        entries = [ln.strip() for ln in f if ln.strip()]

    # 先索引全部 GT 文件，再用 split 里的文件名做匹配，避免相对路径风格不一致导致找不到文件。
    gt_candidates = glob(os.path.join(data_root, "*", "GT", "*_GT.tif"))
    gt_map = {os.path.basename(p): p for p in gt_candidates}
    pairs: List[Tuple[str, str]] = []
    for line in entries:
        gt_name = os.path.basename(line)
        if gt_name not in gt_map:
            raise FileNotFoundError(f"Cannot resolve GT from split line: {line}")
        gt_path = gt_map[gt_name]
        sar_path = gt_path.replace(f"{os.sep}GT{os.sep}", f"{os.sep}SAR{os.sep}").replace("_GT.tif", "_SAR.tif")
        if not os.path.exists(sar_path):
            raise FileNotFoundError(f"SAR not found for GT: {gt_path}")
        pairs.append((sar_path, gt_path))
    return pairs


class UrbanSARFloodsSegDataset(Dataset):
    """UrbanSARFloods 语义分割数据集。

    返回:
        x_dict: {
            imgs: (C,H,W) 归一化影像,
            chn_ids: (C,) 通道语义编码,
            time_ids: (C,) 时相编码
        }
        label: (H,W) 语义标签，非法像素为 ignore_index
        invalid_ratio: 当前样本非法像素比例
    """

    def __init__(
        self,
        data_root: str,
        split_file: str,
        channel_ids: Sequence[int],
        time_ids: Sequence[int],
        mean: Sequence[float],
        std: Sequence[float],
        ignore_index: int = 255,
        random_hflip: bool = False,
        random_vflip: bool = False,
        crop_size: int = 252,
        random_crop: bool = False,
        seed: int = 42,
    ) -> None:
        self.data_root = data_root
        self.pairs = _resolve_paths(data_root, split_file)
        self.channel_ids = torch.tensor(channel_ids, dtype=torch.long)
        self.time_ids = torch.tensor(time_ids, dtype=torch.long)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)
        self.ignore_index = int(ignore_index)
        self.random_hflip = random_hflip
        self.random_vflip = random_vflip
        self.crop_size = crop_size
        self.random_crop = random_crop
        self.seed = int(seed)

        n_chn = len(channel_ids)
        if len(time_ids) != n_chn or len(mean) != n_chn or len(std) != n_chn:
            raise ValueError("channel_ids/time_ids/mean/std length mismatch")

    def __len__(self) -> int:
        return len(self.pairs)

    def _read_pair(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """读取单个样本的 8 通道影像与标签。"""
        sar_path, gt_path = self.pairs[index]
        with rasterio.open(sar_path) as ds:
            imgs = ds.read().astype(np.float32)  # C, H, W
        with rasterio.open(gt_path) as ds:
            label = ds.read(1).astype(np.int64)  # H, W
        return torch.from_numpy(imgs), torch.from_numpy(label)

    def _apply_nan_policy(self, imgs: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """将 NaN/Inf 填充为通道均值，并把对应标签位置置为 ignore_index。"""
        finite = torch.isfinite(imgs)
        valid_mask = finite.all(dim=0)  # H, W

        # 仅做数值稳定处理，不把无效像素当监督信号。
        fill_values = self.mean.expand_as(imgs)
        imgs = torch.where(finite, imgs, fill_values)

        label = label.clone()
        label[~valid_mask] = self.ignore_index
        return imgs, label, valid_mask

    def _maybe_flip(self, imgs: torch.Tensor, label: torch.Tensor, valid_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """随机水平/垂直翻转，训练增强专用。"""
        if self.random_hflip and np.random.rand() < 0.5:
            imgs = torch.flip(imgs, dims=(-1,))
            label = torch.flip(label, dims=(-1,))
            valid_mask = torch.flip(valid_mask, dims=(-1,))
        if self.random_vflip and np.random.rand() < 0.5:
            imgs = torch.flip(imgs, dims=(-2,))
            label = torch.flip(label, dims=(-2,))
            valid_mask = torch.flip(valid_mask, dims=(-2,))
        return imgs, label, valid_mask

    def _maybe_crop(self, imgs: torch.Tensor, label: torch.Tensor, valid_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """按配置执行随机裁剪或中心裁剪。"""
        if self.crop_size is None:
            return imgs, label, valid_mask

        if isinstance(self.crop_size, int):
            ch = cw = self.crop_size
        else:
            ch, cw = self.crop_size
        h, w = label.shape[-2:]
        if ch > h or cw > w:
            raise ValueError(f"crop_size {(ch, cw)} larger than image {(h, w)}")

        if self.random_crop:
            top = int(np.random.randint(0, h - ch + 1))
            left = int(np.random.randint(0, w - cw + 1))
        else:
            top = (h - ch) // 2
            left = (w - cw) // 2

        imgs = imgs[:, top : top + ch, left : left + cw]
        label = label[top : top + ch, left : left + cw]
        valid_mask = valid_mask[top : top + ch, left : left + cw]
        return imgs, label, valid_mask

    def __getitem__(self, index: int):
        """返回训练/验证所需的字典输入、标签和无效像素比例。"""
        imgs, label = self._read_pair(index)
        imgs, label, valid_mask = self._apply_nan_policy(imgs, label)
        imgs, label, valid_mask = self._maybe_crop(imgs, label, valid_mask)
        imgs, label, valid_mask = self._maybe_flip(imgs, label, valid_mask)
        # 通道级标准化，统计量来自配置文件。
        imgs = (imgs - self.mean) / self.std.clamp(min=1e-6)

        x_dict = {
            "imgs": imgs.float(),
            "chn_ids": self.channel_ids.clone(),
            "time_ids": self.time_ids.clone(),
        }
        invalid_ratio = 1.0 - valid_mask.float().mean()
        return x_dict, label.long(), invalid_ratio.float()

    def compute_class_histogram(self, n_classes: int = 3) -> torch.Tensor:
        """统计标签类别频次，用于构造类别权重。"""
        hist = torch.zeros(n_classes, dtype=torch.float64)
        for _, gt_path in self.pairs:
            with rasterio.open(gt_path) as ds:
                label = ds.read(1).astype(np.int64)
            t = torch.from_numpy(label)
            keep = (t >= 0) & (t < n_classes)
            binc = torch.bincount(t[keep].flatten(), minlength=n_classes).double()
            hist += binc
        return hist


def collate_urban_floods(batch):
    """自定义 batch 拼接：将字典字段逐项 stack。"""
    x_dicts, labels, invalid_ratios = zip(*batch)
    out_x: Dict[str, torch.Tensor] = {}
    keys = x_dicts[0].keys()
    for k in keys:
        out_x[k] = torch.stack([x[k] for x in x_dicts], dim=0)
    labels = torch.stack(labels, dim=0)
    invalid_ratios = torch.stack(list(invalid_ratios), dim=0)
    return out_x, labels, invalid_ratios
