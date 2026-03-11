"""监督参考实验的数据读取与 split 管理。"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from .feature_utils import joint_robust_norm, local_change_strength, robust_unit, safe_log_ratio
from .ops import connected_components_with_stats, gradient_magnitude


@dataclass(frozen=True)
class ReferenceTileRecord:
    """单个参考 tile 的配对记录。"""

    sample_id: str
    pre_path: Path
    post_path: Path
    gt_path: Path


@dataclass(frozen=True)
class TargetTileRecord:
    """单个目标 tile 的配对记录。"""

    sample_id: str
    pre_path: Path
    post_path: Path


def _read_split_file(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_split_file(path: Path, items: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(items) + "\n", encoding="utf-8")


def ensure_reference_splits(root: str, split_dir: str, seed: int = 42, val_ratio: float = 0.2) -> Dict[str, Path]:
    """若 split 不存在，则按固定随机种子自动生成。"""
    root_path = Path(root).resolve()
    split_root = Path(split_dir).resolve()
    train_path = split_root / "train.txt"
    val_path = split_root / "val.txt"
    ref_path = split_root / "ref_bank.txt"
    if train_path.exists() and val_path.exists() and ref_path.exists():
        return {"train": train_path, "val": val_path, "ref_bank": ref_path}

    pre_dir = root_path / "pre"
    sample_ids = sorted(p.name.replace("_pre.tif", "") for p in pre_dir.glob("*_pre.tif"))
    if not sample_ids:
        raise FileNotFoundError(f"No pre tiles found under {pre_dir}")
    rng = random.Random(int(seed))
    rng.shuffle(sample_ids)
    val_count = max(1, int(round(len(sample_ids) * float(val_ratio))))
    val_ids = sorted(sample_ids[:val_count])
    train_ids = sorted(sample_ids[val_count:])
    if not train_ids:
        train_ids = val_ids
    ref_ids = list(train_ids)
    _write_split_file(train_path, train_ids)
    _write_split_file(val_path, val_ids)
    _write_split_file(ref_path, ref_ids)
    meta = {
        "seed": int(seed),
        "val_ratio": float(val_ratio),
        "num_train": len(train_ids),
        "num_val": len(val_ids),
        "num_ref": len(ref_ids),
    }
    (split_root / "split_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"train": train_path, "val": val_path, "ref_bank": ref_path}


class UrbanSARReferenceTileDataset(Dataset):
    """读取 pre/post/GT 平铺结构的监督参考 tile。"""

    def __init__(
        self,
        root: str,
        split: str,
        split_dir: str,
        seed: int = 42,
        val_ratio: float = 0.2,
        ignore_index: int = 255,
        max_samples: int = -1,
    ) -> None:
        split_paths = ensure_reference_splits(root, split_dir, seed=seed, val_ratio=val_ratio)
        split_key = split if split in split_paths else Path(split).stem
        split_path = split_paths.get(split_key, Path(split))
        if not Path(split_path).exists():
            raise FileNotFoundError(f"split file not found: {split_path}")
        self.root = Path(root).resolve()
        self.ignore_index = int(ignore_index)
        sample_ids = _read_split_file(Path(split_path))
        self.records = self._build_records(sample_ids)
        if int(max_samples) > 0:
            self.records = self.records[: int(max_samples)]
        self._descriptor_cache: Dict[str, np.ndarray] = {}
        self._positive_ratio_cache: Dict[str, float] = {}

    def _build_records(self, sample_ids: Iterable[str]) -> List[ReferenceTileRecord]:
        records: List[ReferenceTileRecord] = []
        for sample_id in sample_ids:
            pre_path = self.root / "pre" / f"{sample_id}_pre.tif"
            post_path = self.root / "post" / f"{sample_id}_post.tif"
            gt_path = self.root / "GT" / f"{sample_id}_GT.tif"
            missing = [str(p) for p in [pre_path, post_path, gt_path] if not p.exists()]
            if missing:
                raise FileNotFoundError(f"Missing files for {sample_id}: {missing}")
            records.append(ReferenceTileRecord(sample_id=sample_id, pre_path=pre_path, post_path=post_path, gt_path=gt_path))
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict:
        rec = self.records[index]
        with rasterio.open(rec.pre_path) as ds_pre, rasterio.open(rec.post_path) as ds_post, rasterio.open(rec.gt_path) as ds_gt:
            pre = ds_pre.read(1).astype(np.float32)
            post = ds_post.read(1).astype(np.float32)
            gt = ds_gt.read(1)
            if pre.shape != post.shape or pre.shape != gt.shape:
                raise ValueError(f"Shape mismatch for {rec.sample_id}: pre{pre.shape}, post{post.shape}, gt{gt.shape}")
            if ds_pre.crs != ds_post.crs or ds_pre.transform != ds_post.transform:
                raise ValueError(f"Pre/Post georef mismatch for {rec.sample_id}")
            if ds_pre.crs != ds_gt.crs or ds_pre.transform != ds_gt.transform:
                raise ValueError(f"Image/GT georef mismatch for {rec.sample_id}")
            profile = ds_pre.profile.copy()
            transform = ds_pre.transform
            crs = ds_pre.crs

        valid_mask = np.isfinite(pre) & np.isfinite(post)
        maps = build_reference_feature_maps(pre, post, valid_mask, sample_id=rec.sample_id)
        gt = gt.astype(np.int64)
        gt[~valid_mask] = self.ignore_index
        sample = {
            "sample_id": rec.sample_id,
            "pre": torch.from_numpy(maps["pre_norm"].astype(np.float32)),
            "post": torch.from_numpy(maps["post_norm"].astype(np.float32)),
            "diff": torch.from_numpy(maps["diff"].astype(np.float32)),
            "neg_diff": torch.from_numpy(maps["neg_diff"].astype(np.float32)),
            "darkening_score": torch.from_numpy(maps["darkening_score"].astype(np.float32)),
            "brightening_score": torch.from_numpy(maps["brightening_score"].astype(np.float32)),
            "change_score": torch.from_numpy(maps["change_score"].astype(np.float32)),
            "stable_score": torch.from_numpy(maps["stable_score"].astype(np.float32)),
            "boundary_score": torch.from_numpy(maps["boundary_score"].astype(np.float32)),
            "log_ratio_like": torch.from_numpy(maps["log_ratio_like"].astype(np.float32)),
            "local_contrast_score": torch.from_numpy(maps["local_contrast_score"].astype(np.float32)),
            "features": torch.from_numpy(maps["model_input"].astype(np.float32)),
            "pseudo_rgb": torch.from_numpy(maps["pseudo_rgb"].astype(np.uint8)),
            "gt": torch.from_numpy(gt),
            "valid_mask": torch.from_numpy(valid_mask.astype(bool)),
            "meta": {
                "transform": tuple(transform),
                "crs": str(crs),
                "profile": profile,
            },
        }
        return sample

    def get_sample_by_id(self, sample_id: str) -> Dict:
        index = next(i for i, rec in enumerate(self.records) if rec.sample_id == sample_id)
        return self[index]

    def get_positive_ratio(self, sample_id: str) -> float:
        if sample_id in self._positive_ratio_cache:
            return self._positive_ratio_cache[sample_id]
        sample = self.get_sample_by_id(sample_id)
        gt = sample["gt"].numpy()
        valid = gt != self.ignore_index
        ratio = float(((gt > 0) & valid).sum() / max(valid.sum(), 1))
        self._positive_ratio_cache[sample_id] = ratio
        return ratio

    def get_descriptor(self, sample_id: str) -> np.ndarray:
        if sample_id in self._descriptor_cache:
            return self._descriptor_cache[sample_id]
        sample = self.get_sample_by_id(sample_id)
        desc = build_global_descriptor(sample)
        self._descriptor_cache[sample_id] = desc
        return desc

    @property
    def sample_ids(self) -> List[str]:
        return [rec.sample_id for rec in self.records]


class GF3TargetTileDataset(Dataset):
    """读取仅含 pre/post 的无标签目标 tile。"""

    def __init__(
        self,
        root: str,
        max_samples: int = -1,
    ) -> None:
        self.root = Path(root).resolve()
        self.records = self._build_records()
        if int(max_samples) > 0:
            self.records = self.records[: int(max_samples)]

    def _build_records(self) -> List[TargetTileRecord]:
        pre_dir = self.root / "pre"
        post_dir = self.root / "post"
        sample_ids = sorted(p.name.replace("_pre.tif", "") for p in pre_dir.glob("*_pre.tif"))
        records: List[TargetTileRecord] = []
        for sample_id in sample_ids:
            pre_path = pre_dir / f"{sample_id}_pre.tif"
            post_path = post_dir / f"{sample_id}_post.tif"
            missing = [str(p) for p in [pre_path, post_path] if not p.exists()]
            if missing:
                raise FileNotFoundError(f"Missing files for {sample_id}: {missing}")
            records.append(TargetTileRecord(sample_id=sample_id, pre_path=pre_path, post_path=post_path))
        if not records:
            raise FileNotFoundError(f"No target pre/post tiles found under {self.root}")
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict:
        rec = self.records[index]
        with rasterio.open(rec.pre_path) as ds_pre, rasterio.open(rec.post_path) as ds_post:
            pre = ds_pre.read(1).astype(np.float32)
            post = ds_post.read(1).astype(np.float32)
            if pre.shape != post.shape:
                raise ValueError(f"Shape mismatch for {rec.sample_id}: pre{pre.shape}, post{post.shape}")
            if ds_pre.crs != ds_post.crs or ds_pre.transform != ds_post.transform:
                raise ValueError(f"Pre/Post georef mismatch for {rec.sample_id}")
            profile = ds_pre.profile.copy()
            transform = ds_pre.transform
            crs = ds_pre.crs

        valid_mask = np.isfinite(pre) & np.isfinite(post)
        maps = build_reference_feature_maps(pre, post, valid_mask, sample_id=rec.sample_id)
        return {
            "sample_id": rec.sample_id,
            "pre": torch.from_numpy(maps["pre_norm"].astype(np.float32)),
            "post": torch.from_numpy(maps["post_norm"].astype(np.float32)),
            "diff": torch.from_numpy(maps["diff"].astype(np.float32)),
            "neg_diff": torch.from_numpy(maps["neg_diff"].astype(np.float32)),
            "darkening_score": torch.from_numpy(maps["darkening_score"].astype(np.float32)),
            "brightening_score": torch.from_numpy(maps["brightening_score"].astype(np.float32)),
            "change_score": torch.from_numpy(maps["change_score"].astype(np.float32)),
            "stable_score": torch.from_numpy(maps["stable_score"].astype(np.float32)),
            "boundary_score": torch.from_numpy(maps["boundary_score"].astype(np.float32)),
            "log_ratio_like": torch.from_numpy(maps["log_ratio_like"].astype(np.float32)),
            "local_contrast_score": torch.from_numpy(maps["local_contrast_score"].astype(np.float32)),
            "features": torch.from_numpy(maps["model_input"].astype(np.float32)),
            "pseudo_rgb": torch.from_numpy(maps["pseudo_rgb"].astype(np.uint8)),
            "valid_mask": torch.from_numpy(valid_mask.astype(bool)),
            "meta": {
                "transform": tuple(transform),
                "crs": str(crs),
                "profile": profile,
            },
        }

    @property
    def sample_ids(self) -> List[str]:
        return [rec.sample_id for rec in self.records]


class ReferenceQueryPairDataset(Dataset):
    """为监督参考训练按 query 绑定一个参考样本。"""

    def __init__(
        self,
        query_dataset: UrbanSARReferenceTileDataset,
        ref_dataset: UrbanSARReferenceTileDataset,
        seed: int = 42,
    ) -> None:
        self.query_dataset = query_dataset
        self.ref_dataset = ref_dataset
        self.seed = int(seed)
        self.ref_ids = [sid for sid in ref_dataset.sample_ids if ref_dataset.get_positive_ratio(sid) > 0.0]
        if not self.ref_ids:
            self.ref_ids = list(ref_dataset.sample_ids)

    def __len__(self) -> int:
        return len(self.query_dataset)

    def __getitem__(self, index: int) -> Dict:
        query = self.query_dataset[index]
        ref_id = select_reference_id(
            query_sample=query,
            ref_dataset=self.ref_dataset,
            candidate_ids=self.ref_ids,
            exclude_id=query["sample_id"],
            seed=self.seed + index,
        )
        reference = self.ref_dataset.get_sample_by_id(ref_id)
        return {"query": query, "reference": reference}


def build_reference_feature_maps(
    pre: np.ndarray,
    post: np.ndarray,
    valid_mask: np.ndarray,
    sample_id: str | None = None,
) -> Dict[str, np.ndarray]:
    """从配对 SAR tile 构造监督参考实验的特征图。"""
    pre_norm, post_norm = joint_robust_norm(pre, post, valid_mask)
    diff = post_norm - pre_norm
    darkening_raw = np.clip(pre_norm - post_norm, 0.0, None)
    brightening_raw = np.clip(post_norm - pre_norm, 0.0, None)
    neg_diff = darkening_raw.copy()
    log_ratio_like = robust_unit(-safe_log_ratio(pre, post, valid_mask), valid_mask)
    local_contrast_score = robust_unit(local_change_strength(pre, post, valid_mask), valid_mask)
    darkening_score = robust_unit(darkening_raw, valid_mask)
    brightening_score = robust_unit(brightening_raw, valid_mask)
    change_score = np.clip(
        0.45 * darkening_score + 0.30 * log_ratio_like + 0.20 * local_contrast_score + 0.05 * (1.0 - brightening_score),
        0.0,
        1.0,
    )
    stable_score = np.clip(0.65 * (1.0 - change_score) + 0.20 * (1.0 - local_contrast_score) + 0.15 * brightening_score, 0.0, 1.0)
    boundary_score = robust_unit(gradient_magnitude(change_score), valid_mask)
    named_maps = {
        "pre_norm": pre_norm,
        "post_norm": post_norm,
        "diff": diff,
        "neg_diff": neg_diff,
        "darkening_score": darkening_score,
        "brightening_score": brightening_score,
        "log_ratio_like": log_ratio_like,
        "local_contrast_score": local_contrast_score,
        "change_score": change_score,
        "stable_score": stable_score,
        "boundary_score": boundary_score,
    }
    for name, arr in named_maps.items():
        arr[~np.isfinite(arr)] = 0.0
        arr[~valid_mask] = 0.0
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            raise ValueError(f"Non-finite values remain in {name} for sample {sample_id or '<unknown>'}")
    pseudo_rgb = np.stack(
        [
            np.clip(pre_norm * 255.0, 0, 255).astype(np.uint8),
            np.clip(post_norm * 255.0, 0, 255).astype(np.uint8),
            np.clip((0.65 * darkening_score + 0.35 * log_ratio_like) * 255.0, 0, 255).astype(np.uint8),
        ],
        axis=-1,
    )
    pseudo_rgb[~valid_mask] = 0
    model_input = np.stack(
        [
            pre_norm,
            post_norm,
            diff,
            darkening_score,
            brightening_score,
            log_ratio_like,
            local_contrast_score,
            change_score,
            stable_score,
            boundary_score,
        ],
        axis=0,
    ).astype(np.float32)
    if not np.isfinite(model_input).all():
        raise ValueError(f"Non-finite values remain in model_input for sample {sample_id or '<unknown>'}")
    return {
        "pre_norm": pre_norm.astype(np.float32),
        "post_norm": post_norm.astype(np.float32),
        "diff": diff.astype(np.float32),
        "neg_diff": neg_diff.astype(np.float32),
        "darkening_score": darkening_score.astype(np.float32),
        "brightening_score": brightening_score.astype(np.float32),
        "log_ratio_like": log_ratio_like.astype(np.float32),
        "local_contrast_score": local_contrast_score.astype(np.float32),
        "change_score": change_score.astype(np.float32),
        "stable_score": stable_score.astype(np.float32),
        "boundary_score": boundary_score.astype(np.float32),
        "pseudo_rgb": pseudo_rgb,
        "model_input": model_input,
    }


def build_global_descriptor(sample: Dict) -> np.ndarray:
    """构造用于参考样本检索的全局描述子。"""
    valid = sample["valid_mask"].numpy().astype(bool)
    feats = []
    for key in ["pre", "post", "darkening_score", "log_ratio_like", "change_score", "boundary_score", "local_contrast_score"]:
        arr = sample[key].numpy().astype(np.float32)
        vals = arr[valid]
        if vals.size == 0:
            feats.extend([0.0, 0.0, 0.0])
        else:
            feats.extend([float(vals.mean()), float(vals.std()), float(vals.max())])
    pos_mask = (sample["change_score"].numpy() > 0.6) & valid
    num_labels, _, stats = connected_components_with_stats(pos_mask.astype(np.uint8))
    comp_areas = np.asarray([item["area"] for item in stats[1:]], dtype=np.float32)
    feats.extend(
        [
            float(pos_mask.mean()) if valid.any() else 0.0,
            float(num_labels - 1),
            float(comp_areas.max() / max(pos_mask.sum(), 1)) if comp_areas.size > 0 else 0.0,
            float(sample["boundary_score"].numpy()[valid].mean()) if valid.any() else 0.0,
        ]
    )
    return np.asarray(feats, dtype=np.float32)


def select_reference_id(
    query_sample: Dict,
    ref_dataset: UrbanSARReferenceTileDataset,
    candidate_ids: Sequence[str],
    exclude_id: str | None = None,
    seed: int = 42,
) -> str:
    """根据变化统计从参考库中选择最相近的参考样本。"""
    query_desc = build_global_descriptor(query_sample)
    best_id = None
    best_dist = float("inf")
    for sample_id in candidate_ids:
        if exclude_id and sample_id == exclude_id and len(candidate_ids) > 1:
            continue
        ref_desc = ref_dataset.get_descriptor(sample_id)
        diff = np.abs(query_desc - ref_desc)
        stats_dist = float(np.mean(diff[: 7 * 3]))
        morph_dist = float(np.mean(diff[7 * 3 :]))
        area_dist = abs(ref_dataset.get_positive_ratio(sample_id) - _query_positive_proxy(query_sample))
        dist = 0.55 * stats_dist + 0.25 * area_dist + 0.20 * morph_dist
        if dist < best_dist:
            best_dist = dist
            best_id = sample_id
    if best_id is not None:
        return best_id
    rng = random.Random(int(seed))
    return rng.choice(list(candidate_ids))


def _query_positive_proxy(sample: Dict) -> float:
    change_score = sample["change_score"].numpy().astype(np.float32)
    valid = sample["valid_mask"].numpy().astype(bool)
    if not valid.any():
        return 0.0
    threshold = max(float(np.quantile(change_score[valid], 0.9)), 0.4)
    return float(((change_score >= threshold) & valid).sum() / valid.sum())
