"""GF3_Henan 单极化双时相变化先验驱动的 prompt 生成。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from .gf3_preprocess import robust_unit
from .ops import binary_dilate, binary_open, connected_components_with_stats, gradient_magnitude


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
    stable_score: np.ndarray
    boundary_score: np.ndarray
    pseudo_rgb: np.ndarray
    valid_mask: np.ndarray
    low_confidence: bool


class PromptCandidateGenerator:
    """面向 GF3 双时相 tile 生成正负候选点。"""

    def __init__(
        self,
        patch_radius: int = 6,
        positive_quantile: float = 0.92,
        negative_quantile: float = 0.35,
        min_component_area: int = 24,
        min_point_distance: int = 16,
        max_positive_candidates: int = 18,
        max_negative_candidates: int = 18,
        initial_positive_points: int = 6,
        initial_negative_points: int = 6,
        low_confidence_max_score: float = 0.18,
    ) -> None:
        self.patch_radius = int(patch_radius)
        self.positive_quantile = float(positive_quantile)
        self.negative_quantile = float(negative_quantile)
        self.min_component_area = int(min_component_area)
        self.min_point_distance = int(min_point_distance)
        self.max_positive_candidates = int(max_positive_candidates)
        self.max_negative_candidates = int(max_negative_candidates)
        self.initial_positive_points = int(initial_positive_points)
        self.initial_negative_points = int(initial_negative_points)
        self.low_confidence_max_score = float(low_confidence_max_score)

    def generate(self, sample: Dict) -> PromptSet:
        """从 GF3 tile 样本构造 prompt set。"""
        pre = sample["pre"].detach().cpu().numpy().astype(np.float32)
        post = sample["post"].detach().cpu().numpy().astype(np.float32)
        diff = sample["diff"].detach().cpu().numpy().astype(np.float32)
        change_score = sample["change_score"].detach().cpu().numpy().astype(np.float32)
        log_ratio_like = sample["log_ratio_like"].detach().cpu().numpy().astype(np.float32)
        valid_mask = sample["valid_mask"].detach().cpu().numpy().astype(bool)

        stable_score = np.clip(0.8 * (1.0 - change_score) + 0.2 * (1.0 - np.abs(diff)), 0.0, 1.0)
        stable_score[~valid_mask] = 0.0
        boundary_score = robust_unit(gradient_magnitude(change_score), valid_mask)
        pseudo_rgb = np.stack([
            np.clip(pre * 255.0, 0, 255).astype(np.uint8),
            np.clip(post * 255.0, 0, 255).astype(np.uint8),
            np.clip(change_score * 255.0, 0, 255).astype(np.uint8),
        ], axis=-1)
        pseudo_rgb[~valid_mask] = 0

        maps = {
            "pre": pre,
            "post": post,
            "diff": diff,
            "change_score": change_score,
            "stable_score": stable_score,
            "boundary_score": boundary_score,
            "log_ratio_like": log_ratio_like,
        }
        pos_nodes = self._generate_positive_nodes(maps, valid_mask)
        neg_nodes = self._generate_negative_nodes(maps, valid_mask, pos_nodes)
        low_conf = bool(float(change_score[valid_mask].max()) < self.low_confidence_max_score) if valid_mask.any() else True

        return PromptSet(
            positive_nodes=pos_nodes,
            negative_nodes=neg_nodes,
            initial_positive_ids=[n.node_id for n in pos_nodes[: self.initial_positive_points]],
            initial_negative_ids=[n.node_id for n in neg_nodes[: self.initial_negative_points]],
            change_score=change_score,
            stable_score=stable_score,
            boundary_score=boundary_score,
            pseudo_rgb=pseudo_rgb,
            valid_mask=valid_mask,
            low_confidence=low_conf,
        )

    def _generate_positive_nodes(self, maps: Dict[str, np.ndarray], valid_mask: np.ndarray) -> List[CandidateNode]:
        score = maps["change_score"]
        q = _safe_quantile(score[valid_mask], self.positive_quantile, fallback=0.75)
        q = max(q, 0.35)
        binary = (score >= q).astype(np.uint8)
        binary[~valid_mask] = 0
        binary = binary_open(binary, 3)
        num_labels, labels, stats = connected_components_with_stats(binary)

        nodes: List[CandidateNode] = []
        next_id = 0
        for comp_id in range(1, num_labels):
            if int(stats[comp_id]["area"]) < self.min_component_area:
                continue
            comp_mask = labels == comp_id
            for y, x in _topk_coords(score * comp_mask, k=3):
                if not valid_mask[y, x]:
                    continue
                node = CandidateNode(next_id, "pos", int(y), int(x), float(score[y, x]), extract_node_feature(maps, y, x, self.patch_radius))
                nodes.append(node)
                next_id += 1

        for y, x in _topk_coords(score, k=self.max_positive_candidates * 6):
            if not valid_mask[y, x]:
                continue
            node = CandidateNode(next_id, "pos", int(y), int(x), float(score[y, x]), extract_node_feature(maps, y, x, self.patch_radius))
            if _accept_node(nodes, node, self.min_point_distance):
                nodes.append(node)
                next_id += 1
            if len(nodes) >= self.max_positive_candidates:
                break
        return _deduplicate_and_sort(nodes, self.min_point_distance, self.max_positive_candidates)

    def _generate_negative_nodes(self, maps: Dict[str, np.ndarray], valid_mask: np.ndarray, pos_nodes: Sequence[CandidateNode]) -> List[CandidateNode]:
        stable_score = maps["stable_score"]
        boundary_score = maps["boundary_score"]
        change_score = maps["change_score"]
        pos_mask = np.zeros_like(valid_mask, dtype=np.uint8)
        for node in pos_nodes:
            pos_mask[node.y, node.x] = 1
        ring = (binary_dilate(pos_mask, 17, iterations=1) > 0) & (~pos_mask.astype(bool)) & valid_mask
        neg_score = np.clip(0.75 * stable_score + 0.25 * boundary_score, 0.0, 1.0)
        neg_score[change_score >= max(_safe_quantile(change_score[valid_mask], self.positive_quantile, fallback=0.75), 0.35)] = 0.0

        nodes: List[CandidateNode] = []
        next_id = 0
        for source_mask in [ring, valid_mask]:
            score_map = neg_score * source_mask.astype(np.float32)
            for y, x in _topk_coords(score_map, k=self.max_negative_candidates * 8):
                if not valid_mask[y, x]:
                    continue
                node = CandidateNode(next_id, "neg", int(y), int(x), float(score_map[y, x]), extract_node_feature(maps, y, x, self.patch_radius))
                if _accept_node(nodes, node, self.min_point_distance):
                    nodes.append(node)
                    next_id += 1
                if len(nodes) >= self.max_negative_candidates:
                    break
            if len(nodes) >= self.max_negative_candidates:
                break
        return _deduplicate_and_sort(nodes, self.min_point_distance, self.max_negative_candidates)


def extract_node_feature(maps: Dict[str, np.ndarray], y: int, x: int, radius: int) -> Tuple[float, ...]:
    """提取局部 patch 统计特征。"""
    h, w = maps["change_score"].shape
    top = max(0, y - radius)
    bottom = min(h, y + radius + 1)
    left = max(0, x - radius)
    right = min(w, x + radius + 1)
    pre_patch = maps["pre"][top:bottom, left:right]
    post_patch = maps["post"][top:bottom, left:right]
    diff_patch = maps["diff"][top:bottom, left:right]
    score_patch = maps["change_score"][top:bottom, left:right]
    boundary_patch = maps["boundary_score"][top:bottom, left:right]
    feat = (
        float(pre_patch.mean()),
        float(pre_patch.std()),
        float(post_patch.mean()),
        float(post_patch.std()),
        float(diff_patch.mean()),
        float(diff_patch.std()),
        float(score_patch.mean()),
        float(score_patch.max()),
        float(boundary_patch.mean()),
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
