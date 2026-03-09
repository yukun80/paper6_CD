from typing import Optional

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

from data.dataset import UrbanSARFloodsDataset


def build_train_sampler(
    dataset: UrbanSARFloodsDataset,
    enable_urban_oversample: bool = True,
    urban_weight: float = 4.0,
    open_weight: float = 1.5,
    hard_scores: Optional[np.ndarray] = None,
) -> Optional[WeightedRandomSampler]:
    """构建面向 urban 长尾的加权采样器。"""
    if not enable_urban_oversample and hard_scores is None:
        return None

    weights = np.ones(len(dataset), dtype=np.float64)

    if enable_urban_oversample:
        for i, st in enumerate(dataset.sample_stats):
            # urban像素占比越高，权重越高；open 次之。
            w = 1.0
            w += urban_weight * st["urban_ratio"]
            w += open_weight * st["open_ratio"]
            if st["has_urban"] > 0:
                w *= 1.3
            weights[i] = w

    if hard_scores is not None:
        hard_scores = np.asarray(hard_scores, dtype=np.float64)
        if hard_scores.shape[0] != len(weights):
            raise ValueError("hard_scores length mismatch")
        weights *= (1.0 + np.clip(hard_scores, 0.0, None))

    weights = np.clip(weights, 1e-6, None)
    sampler = WeightedRandomSampler(torch.from_numpy(weights).double(), num_samples=len(weights), replacement=True)
    return sampler
