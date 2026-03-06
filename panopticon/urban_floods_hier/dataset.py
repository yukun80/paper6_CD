import os
from glob import glob
from typing import Dict, List, Sequence, Tuple

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

# 默认 12 通道元信息定义表（按输入通道顺序）。
# feature_type_id: 0=coherence, 1=intensity, 2=delta_coherence, 3=delta_intensity
# temporal_role_id: 0=pre, 1=co, 2=post, 3=pre_minus_co, 4=post_minus_pre
# polarization_id: 0=VV, 1=VH
DEFAULT_CHANNEL_NAMES = [
    "coh_pre_vh",
    "coh_pre_vv",
    "coh_co_vh",
    "coh_co_vv",
    "int_pre_vh",
    "int_pre_vv",
    "int_co_vh",
    "int_co_vv",
    "dcoh_vh",
    "dcoh_vv",
    "dint_vh",
    "dint_vv",
]
DEFAULT_FEATURE_TYPE_IDS = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3]
DEFAULT_TEMPORAL_ROLE_IDS = [0, 0, 1, 1, 0, 0, 1, 1, 3, 3, 4, 4]
DEFAULT_POLARIZATION_IDS = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]


def _resolve_split_entries(data_root: str, split_file: str) -> List[str]:
    split_path = split_file if os.path.isabs(split_file) else os.path.join(data_root, split_file)
    with open(split_path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def _replace_dir(path: str, src_dir: str, dst_dir: str) -> str:
    return path.replace(f"{os.sep}{src_dir}{os.sep}", f"{os.sep}{dst_dir}{os.sep}")


def _try_resolve_hier_label(main_gt_path: str, src_dir: str, dst_dir: str, suffix_tag: str) -> str:
    """基于主标签路径解析层次标签路径，兼容多种命名风格。"""
    same_name = _replace_dir(main_gt_path, src_dir, dst_dir)
    if os.path.exists(same_name):
        return same_name

    dirname = os.path.dirname(same_name)
    basename = os.path.basename(main_gt_path)
    candidates = [basename]
    if basename.endswith("_GT.tif"):
        root = basename[: -len("_GT.tif")]
        candidates.append(f"{root}_GT_{suffix_tag}.tif")
        candidates.append(f"{root}_{suffix_tag}.tif")

    for name in candidates:
        p = os.path.join(dirname, name)
        if os.path.exists(p):
            return p

    raise FileNotFoundError(
        f"Cannot resolve hierarchical label for {main_gt_path}. "
        f"Tried dir={dst_dir} with candidates={candidates}"
    )


def _resolve_paths(
    data_root: str,
    split_file: str,
    main_label_dir: str,
    sar_dir: str,
    floodness_label_dir: str,
    flood_type_label_dir: str,
) -> List[Tuple[str, str, str, str]]:
    entries = _resolve_split_entries(data_root, split_file)

    # 先索引主标签，避免 split 中相对路径风格差异造成找不到文件。
    gt_candidates = glob(os.path.join(data_root, "*", main_label_dir, "*_GT.tif"))
    gt_map = {os.path.basename(p): p for p in gt_candidates}

    pairs: List[Tuple[str, str, str, str]] = []
    for line in entries:
        gt_name = os.path.basename(line)
        if gt_name not in gt_map:
            raise FileNotFoundError(f"Cannot resolve main GT from split line: {line}")

        main_gt_path = gt_map[gt_name]
        sar_path = _replace_dir(main_gt_path, main_label_dir, sar_dir).replace("_GT.tif", "_SAR.tif")
        if not os.path.exists(sar_path):
            raise FileNotFoundError(f"SAR not found for main GT: {main_gt_path}")

        floodness_path = _try_resolve_hier_label(
            main_gt_path,
            src_dir=main_label_dir,
            dst_dir=floodness_label_dir,
            suffix_tag="floodness",
        )
        flood_type_path = _try_resolve_hier_label(
            main_gt_path,
            src_dir=main_label_dir,
            dst_dir=flood_type_label_dir,
            suffix_tag="flood_type",
        )
        pairs.append((sar_path, main_gt_path, floodness_path, flood_type_path))

    return pairs


class UrbanSARFloodsHierDataset(Dataset):
    """UrbanSARFloods 层次化分割数据集。

    返回:
        x_dict: {
            imgs, chn_ids, time_ids,
            feature_type_ids, temporal_role_ids, polarization_ids
        }
        y_dict: {
            main_label, floodness_label, flood_type_label
        }
        invalid_ratio: 当前样本无效像素比例
    """

    def __init__(
        self,
        data_root: str,
        split_file: str,
        channel_ids: Sequence[int],
        time_ids: Sequence[int],
        feature_type_ids: Sequence[int],
        temporal_role_ids: Sequence[int],
        polarization_ids: Sequence[int],
        mean: Sequence[float],
        std: Sequence[float],
        main_label_dir: str = "GT",
        floodness_label_dir: str = "GT_floodness",
        flood_type_label_dir: str = "GT_flood_type",
        sar_dir: str = "SAR",
        ignore_index_main: int = 255,
        ignore_index_floodness: int = 255,
        ignore_index_flood_type: int = 255,
        random_hflip: bool = False,
        random_vflip: bool = False,
        crop_size: int = 252,
        random_crop: bool = False,
        seed: int = 42,
    ) -> None:
        self.data_root = data_root
        self.pairs = _resolve_paths(
            data_root=data_root,
            split_file=split_file,
            main_label_dir=main_label_dir,
            sar_dir=sar_dir,
            floodness_label_dir=floodness_label_dir,
            flood_type_label_dir=flood_type_label_dir,
        )

        self.channel_ids = torch.tensor(channel_ids, dtype=torch.long)
        self.time_ids = torch.tensor(time_ids, dtype=torch.long)
        self.feature_type_ids = torch.tensor(feature_type_ids, dtype=torch.long)
        self.temporal_role_ids = torch.tensor(temporal_role_ids, dtype=torch.long)
        self.polarization_ids = torch.tensor(polarization_ids, dtype=torch.long)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)

        self.ignore_index_main = int(ignore_index_main)
        self.ignore_index_floodness = int(ignore_index_floodness)
        self.ignore_index_flood_type = int(ignore_index_flood_type)
        self.random_hflip = random_hflip
        self.random_vflip = random_vflip
        self.crop_size = crop_size
        self.random_crop = random_crop
        self.seed = int(seed)

        n_chn = len(channel_ids)
        lengths = [
            len(time_ids),
            len(feature_type_ids),
            len(temporal_role_ids),
            len(polarization_ids),
            len(mean),
            len(std),
        ]
        if any(x != n_chn for x in lengths):
            raise ValueError(f"Channel metadata length mismatch: expected {n_chn}, got {lengths}")

    def __len__(self) -> int:
        return len(self.pairs)

    def _read_pair(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """读取 12 通道影像和三路标签。"""
        sar_path, main_gt_path, floodness_path, flood_type_path = self.pairs[index]
        with rasterio.open(sar_path) as ds:
            imgs = ds.read().astype(np.float32)
        with rasterio.open(main_gt_path) as ds:
            main_label = ds.read(1).astype(np.int64)
        with rasterio.open(floodness_path) as ds:
            floodness_label = ds.read(1).astype(np.int64)
        with rasterio.open(flood_type_path) as ds:
            flood_type_label = ds.read(1).astype(np.int64)

        return (
            torch.from_numpy(imgs),
            torch.from_numpy(main_label),
            torch.from_numpy(floodness_label),
            torch.from_numpy(flood_type_label),
        )

    def _apply_nan_policy(
        self,
        imgs: torch.Tensor,
        main_label: torch.Tensor,
        floodness_label: torch.Tensor,
        flood_type_label: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """无效像素数值稳定处理，并同步标注 ignore。"""
        finite = torch.isfinite(imgs)
        valid_mask = finite.all(dim=0)

        fill_values = self.mean.expand_as(imgs)
        imgs = torch.where(finite, imgs, fill_values)

        main_label = main_label.clone()
        floodness_label = floodness_label.clone()
        flood_type_label = flood_type_label.clone()
        main_label[~valid_mask] = self.ignore_index_main
        floodness_label[~valid_mask] = self.ignore_index_floodness
        flood_type_label[~valid_mask] = self.ignore_index_flood_type
        return imgs, main_label, floodness_label, flood_type_label, valid_mask

    def _maybe_flip(
        self,
        imgs: torch.Tensor,
        main_label: torch.Tensor,
        floodness_label: torch.Tensor,
        flood_type_label: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """随机翻转增强。"""
        if self.random_hflip and np.random.rand() < 0.5:
            imgs = torch.flip(imgs, dims=(-1,))
            main_label = torch.flip(main_label, dims=(-1,))
            floodness_label = torch.flip(floodness_label, dims=(-1,))
            flood_type_label = torch.flip(flood_type_label, dims=(-1,))
            valid_mask = torch.flip(valid_mask, dims=(-1,))

        if self.random_vflip and np.random.rand() < 0.5:
            imgs = torch.flip(imgs, dims=(-2,))
            main_label = torch.flip(main_label, dims=(-2,))
            floodness_label = torch.flip(floodness_label, dims=(-2,))
            flood_type_label = torch.flip(flood_type_label, dims=(-2,))
            valid_mask = torch.flip(valid_mask, dims=(-2,))

        return imgs, main_label, floodness_label, flood_type_label, valid_mask

    def _maybe_crop(
        self,
        imgs: torch.Tensor,
        main_label: torch.Tensor,
        floodness_label: torch.Tensor,
        flood_type_label: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """随机裁剪或中心裁剪。"""
        if self.crop_size is None:
            return imgs, main_label, floodness_label, flood_type_label, valid_mask

        if isinstance(self.crop_size, int):
            ch = cw = self.crop_size
        else:
            ch, cw = self.crop_size

        h, w = main_label.shape[-2:]
        if ch > h or cw > w:
            raise ValueError(f"crop_size {(ch, cw)} larger than image {(h, w)}")

        if self.random_crop:
            top = int(np.random.randint(0, h - ch + 1))
            left = int(np.random.randint(0, w - cw + 1))
        else:
            top = (h - ch) // 2
            left = (w - cw) // 2

        imgs = imgs[:, top : top + ch, left : left + cw]
        main_label = main_label[top : top + ch, left : left + cw]
        floodness_label = floodness_label[top : top + ch, left : left + cw]
        flood_type_label = flood_type_label[top : top + ch, left : left + cw]
        valid_mask = valid_mask[top : top + ch, left : left + cw]
        return imgs, main_label, floodness_label, flood_type_label, valid_mask

    def __getitem__(self, index: int):
        imgs, main_label, floodness_label, flood_type_label = self._read_pair(index)
        imgs, main_label, floodness_label, flood_type_label, valid_mask = self._apply_nan_policy(
            imgs,
            main_label,
            floodness_label,
            flood_type_label,
        )
        imgs, main_label, floodness_label, flood_type_label, valid_mask = self._maybe_crop(
            imgs,
            main_label,
            floodness_label,
            flood_type_label,
            valid_mask,
        )
        imgs, main_label, floodness_label, flood_type_label, valid_mask = self._maybe_flip(
            imgs,
            main_label,
            floodness_label,
            flood_type_label,
            valid_mask,
        )

        imgs = (imgs - self.mean) / self.std.clamp(min=1e-6)

        x_dict = {
            "imgs": imgs.float(),
            "chn_ids": self.channel_ids.clone(),
            "time_ids": self.time_ids.clone(),
            "feature_type_ids": self.feature_type_ids.clone(),
            "temporal_role_ids": self.temporal_role_ids.clone(),
            "polarization_ids": self.polarization_ids.clone(),
        }
        y_dict = {
            "main_label": main_label.long(),
            "floodness_label": floodness_label.long(),
            "flood_type_label": flood_type_label.long(),
            "valid_mask": valid_mask.bool(),
        }
        invalid_ratio = 1.0 - valid_mask.float().mean()
        return x_dict, y_dict, invalid_ratio.float()

    def compute_main_class_histogram(self, n_classes: int = 3) -> torch.Tensor:
        """统计主标签类别频次，用于自动类别权重。"""
        hist = torch.zeros(n_classes, dtype=torch.float64)
        for _, main_gt_path, _, _ in self.pairs:
            with rasterio.open(main_gt_path) as ds:
                label = ds.read(1).astype(np.int64)
            t = torch.from_numpy(label)
            keep = (t >= 0) & (t < n_classes)
            binc = torch.bincount(t[keep].flatten(), minlength=n_classes).double()
            hist += binc
        return hist


def collate_urban_floods_hier(batch):
    """自定义 batch 拼接：x_dict/y_dict 分别逐项 stack。"""
    x_dicts, y_dicts, invalid_ratios = zip(*batch)

    out_x: Dict[str, torch.Tensor] = {}
    for k in x_dicts[0].keys():
        out_x[k] = torch.stack([x[k] for x in x_dicts], dim=0)

    out_y: Dict[str, torch.Tensor] = {}
    for k in y_dicts[0].keys():
        out_y[k] = torch.stack([y[k] for y in y_dicts], dim=0)

    invalid_ratios = torch.stack(list(invalid_ratios), dim=0)
    return out_x, out_y, invalid_ratios
