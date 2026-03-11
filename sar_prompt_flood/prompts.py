"""监督参考实验复用的 prompt 类型与工具函数。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class CandidateNode:
    """候选点节点。"""

    node_id: int
    label: str
    y: int
    x: int
    score: float
    feature: Tuple[float, ...]


@dataclass
class PromptSet:
    """候选点池、先验图和初始 prompt。"""

    positive_nodes: List[CandidateNode]
    negative_nodes: List[CandidateNode]
    initial_positive_ids: List[int]
    initial_negative_ids: List[int]
    change_score: np.ndarray
    darkening_score: np.ndarray
    log_ratio_score: np.ndarray
    stable_score: np.ndarray
    boundary_score: np.ndarray
    pseudo_rgb: np.ndarray
    valid_mask: np.ndarray
    low_confidence: bool
    area_prior: float


def extract_node_feature(maps: Dict[str, np.ndarray], y: int, x: int, radius: int) -> Tuple[float, ...]:
    """提取局部 patch 统计特征。"""
    h, w = maps["change_score"].shape
    top = max(0, y - radius)
    bottom = min(h, y + radius + 1)
    left = max(0, x - radius)
    right = min(w, x + radius + 1)
    pre_patch = maps["pre"][top:bottom, left:right]
    post_patch = maps["post"][top:bottom, left:right]
    dark_patch = maps["darkening_score"][top:bottom, left:right]
    log_ratio_patch = maps["log_ratio_like"][top:bottom, left:right]
    local_patch = maps["local_contrast_score"][top:bottom, left:right]
    stable_patch = maps["stable_score"][top:bottom, left:right]
    boundary_patch = maps["boundary_score"][top:bottom, left:right]
    feat = (
        float(pre_patch.mean()),
        float(pre_patch.std()),
        float(post_patch.mean()),
        float(post_patch.std()),
        float(dark_patch.mean()),
        float(dark_patch.max()),
        float(log_ratio_patch.mean()),
        float(log_ratio_patch.max()),
        float(local_patch.mean()),
        float(boundary_patch.mean()),
        float(stable_patch.mean()),
        float(y / max(h - 1, 1)),
        float(x / max(w - 1, 1)),
    )
    return feat


def _safe_quantile(values: np.ndarray, q: float, fallback: float) -> float:
    if values.size == 0:
        return float(fallback)
    return float(np.quantile(values, q))


def _topk_coords(score_map: np.ndarray, k: int) -> List[Tuple[int, int]]:
    flat = score_map.reshape(-1)
    if flat.size == 0:
        return []
    k = min(int(k), flat.size)
    if k <= 0:
        return []
    idx = np.argpartition(flat, -k)[-k:]
    idx = idx[np.argsort(flat[idx])[::-1]]
    w = score_map.shape[1]
    return [(int(i // w), int(i % w)) for i in idx]


def _accept_node(existing: Sequence[CandidateNode], node: CandidateNode, min_distance: int) -> bool:
    for other in existing:
        if (other.y - node.y) ** 2 + (other.x - node.x) ** 2 < min_distance ** 2:
            return False
    return True


def _deduplicate_and_sort(nodes: Sequence[CandidateNode], min_distance: int, max_points: int) -> List[CandidateNode]:
    out: List[CandidateNode] = []
    for node in sorted(nodes, key=lambda x: x.score, reverse=True):
        if _accept_node(out, node, min_distance):
            out.append(node)
        if len(out) >= max_points:
            break
    return out
