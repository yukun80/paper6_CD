"""区域生长与连通域后处理。"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy import ndimage


def get_neighbor_offsets(connectivity: int = 8) -> List[Tuple[int, int]]:
    if connectivity == 4:
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]
    return [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]


def region_grow_from_seeds(seed_mask: np.ndarray, candidate_mask: np.ndarray, connectivity: int = 8) -> np.ndarray:
    """使用形态学重建做区域生长，只允许 seed 可达区域进入结果。"""

    if seed_mask.shape != candidate_mask.shape:
        raise ValueError("seed_mask 与 candidate_mask 尺寸不一致")

    structure = ndimage.generate_binary_structure(2, 2 if connectivity == 8 else 1)
    marker = np.asarray(seed_mask & candidate_mask, dtype=bool)
    mask = np.asarray(candidate_mask, dtype=bool)
    return ndimage.binary_propagation(marker, structure=structure, mask=mask).astype(bool)


def remove_small_components(mask: np.ndarray, min_pixels: int, connectivity: int = 8) -> np.ndarray:
    if min_pixels <= 1:
        return mask.copy()
    structure = ndimage.generate_binary_structure(2, 2 if connectivity == 8 else 1)
    labeled, num = ndimage.label(mask, structure=structure)
    if num == 0:
        return mask.copy()
    counts = np.bincount(labeled.ravel())
    keep = counts >= int(min_pixels)
    keep[0] = False
    return keep[labeled]
