"""GF3 无标注 prompt 优化环境与规则优化器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .ops import binary_dilate, binary_erode, connected_components_with_stats
from .prompts import CandidateNode, PromptSet
from .segmenter import BasePromptSegmenter


ACTIONS: Tuple[str, ...] = (
    "drop_pos_worst",
    "drop_neg_worst",
    "restore_pos_best",
    "restore_neg_best",
    "swap_pos_best",
    "swap_neg_best",
    "swap_pos_diverse",
    "swap_neg_boundary",
    "inject_boundary_pos",
    "inject_hard_negative",
)

DEFAULT_OBJECTIVE_WEIGHTS: Dict[str, float] = {
    "darkening_support": 0.22,
    "log_ratio_support": 0.08,
    "outside_contrast": 0.18,
    "boundary_alignment": 0.14,
    "component_quality": 0.10,
    "shape_spread": 0.07,
    "neg_ring_score": 0.08,
    "cross_sep": 0.08,
    "mean_pos_score": 0.05,
    "mean_neg_score": 0.04,
    "redundancy": -0.08,
    "area_penalty": -0.10,
}


@dataclass
class OptimizationSummary:
    """单个 tile 的优化结果。"""

    pos_points: List[List[int]]
    neg_points: List[List[int]]
    mask: np.ndarray
    metrics: Dict[str, float]
    action_history: List[str]


class PromptOptimizationEnv:
    """在固定候选池上做无标注 prompt 子集优化。"""

    def __init__(
        self,
        prompt_set: PromptSet,
        segmenter: BasePromptSegmenter,
        max_steps: int = 10,
        min_positive_points: int = 2,
        max_positive_points: int = 12,
        min_negative_points: int = 2,
        max_negative_points: int = 12,
        objective_weights: Dict[str, float] | None = None,
    ) -> None:
        self.prompt_set = prompt_set
        self.segmenter = segmenter
        self.max_steps = int(max_steps)
        self.min_positive_points = int(min_positive_points)
        self.max_positive_points = int(max_positive_points)
        self.min_negative_points = int(min_negative_points)
        self.max_negative_points = int(max_negative_points)
        self.pos_nodes = {node.node_id: node for node in prompt_set.positive_nodes}
        self.neg_nodes = {node.node_id: node for node in prompt_set.negative_nodes}
        self.pos_sorted = sorted(prompt_set.positive_nodes, key=lambda x: x.score, reverse=True)
        self.neg_sorted = sorted(prompt_set.negative_nodes, key=lambda x: x.score, reverse=True)
        self.objective_weights = {**DEFAULT_OBJECTIVE_WEIGHTS, **(objective_weights or {})}
        self.eval_cache: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], Tuple[float, Dict[str, float], np.ndarray]] = {}
        self.reset()

    def reset(self):
        self.selected_pos = set(self.prompt_set.initial_positive_ids)
        self.selected_neg = set(self.prompt_set.initial_negative_ids)
        self.steps = 0
        self.action_history: List[str] = []
        self.current_objective, self.current_metrics, self.current_mask = self._evaluate_selection(self.selected_pos, self.selected_neg)
        return self

    def get_available_actions(self) -> List[str]:
        actions: List[str] = []
        if len(self.selected_pos) > self.min_positive_points:
            actions.append("drop_pos_worst")
        if len(self.selected_neg) > self.min_negative_points:
            actions.append("drop_neg_worst")
        if len(self.selected_pos) < self.max_positive_points and len(self.selected_pos) < len(self.pos_nodes):
            actions.extend(["restore_pos_best", "swap_pos_best", "swap_pos_diverse", "inject_boundary_pos"])
        if len(self.selected_neg) < self.max_negative_points and len(self.selected_neg) < len(self.neg_nodes):
            actions.extend(["restore_neg_best", "swap_neg_best", "swap_neg_boundary", "inject_hard_negative"])
        if len(self.selected_pos) >= self.max_positive_points:
            actions.extend(["swap_pos_best", "swap_pos_diverse", "inject_boundary_pos"])
        if len(self.selected_neg) >= self.max_negative_points:
            actions.extend(["swap_neg_best", "swap_neg_boundary", "inject_hard_negative"])
        return [a for a in ACTIONS if a in actions]

    def evaluate_action_delta(self, action: str) -> Tuple[float, Dict[str, float], np.ndarray]:
        pos_sel = set(self.selected_pos)
        neg_sel = set(self.selected_neg)
        changed = self._apply_action_to_sets(action, pos_sel, neg_sel)
        if not changed:
            return -1e9, dict(self.current_metrics), self.current_mask.copy()
        objective, metrics, mask = self._evaluate_selection(pos_sel, neg_sel)
        return objective - self.current_objective, metrics, mask

    def apply_action(self, action: str) -> bool:
        changed = self._apply_action_to_sets(action, self.selected_pos, self.selected_neg)
        if not changed:
            return False
        self.current_objective, self.current_metrics, self.current_mask = self._evaluate_selection(self.selected_pos, self.selected_neg)
        self.action_history.append(action)
        self.steps += 1
        return True

    def export_summary(self) -> OptimizationSummary:
        pos_points = [[node.x, node.y] for node in self._selected_nodes(self.pos_nodes, self.selected_pos)]
        neg_points = [[node.x, node.y] for node in self._selected_nodes(self.neg_nodes, self.selected_neg)]
        return OptimizationSummary(
            pos_points=pos_points,
            neg_points=neg_points,
            mask=self.current_mask.copy(),
            metrics=dict(self.current_metrics),
            action_history=list(self.action_history),
        )

    def _apply_action_to_sets(self, action: str, pos_sel: set[int], neg_sel: set[int]) -> bool:
        if action == "drop_pos_worst":
            node = self._worst_selected(self.pos_nodes, pos_sel)
            if node is None:
                return False
            pos_sel.remove(node.node_id)
            return True
        if action == "drop_neg_worst":
            node = self._worst_selected(self.neg_nodes, neg_sel)
            if node is None:
                return False
            neg_sel.remove(node.node_id)
            return True
        if action == "restore_pos_best":
            node = self._best_reserve(self.pos_sorted, pos_sel)
            if node is None:
                return False
            pos_sel.add(node.node_id)
            return True
        if action == "restore_neg_best":
            node = self._best_reserve(self.neg_sorted, neg_sel)
            if node is None:
                return False
            neg_sel.add(node.node_id)
            return True
        if action == "swap_pos_best":
            return self._swap_best(self.pos_sorted, self.pos_nodes, pos_sel)
        if action == "swap_neg_best":
            return self._swap_best(self.neg_sorted, self.neg_nodes, neg_sel)
        if action == "swap_pos_diverse":
            return self._swap_diverse(self.pos_sorted, self.pos_nodes, pos_sel)
        if action == "swap_neg_boundary":
            node = self._boundary_reserve(self.neg_sorted, neg_sel)
            worst = self._worst_selected(self.neg_nodes, neg_sel)
            if node is None or worst is None or node.node_id == worst.node_id:
                return False
            neg_sel.remove(worst.node_id)
            neg_sel.add(node.node_id)
            return True
        if action == "inject_boundary_pos":
            node = self._boundary_reserve(self.pos_sorted, pos_sel)
            return self._inject_with_replacement(node, self.pos_nodes, pos_sel, self.max_positive_points)
        if action == "inject_hard_negative":
            node = self._hard_negative_reserve(neg_sel)
            return self._inject_with_replacement(node, self.neg_nodes, neg_sel, self.max_negative_points)
        return False

    def _inject_with_replacement(self, node: CandidateNode | None, table: Dict[int, CandidateNode], selected: set[int], limit: int) -> bool:
        if node is None:
            return False
        if node.node_id in selected:
            return False
        if len(selected) < limit:
            selected.add(node.node_id)
            return True
        worst = self._worst_selected(table, selected)
        if worst is None or worst.node_id == node.node_id:
            return False
        selected.remove(worst.node_id)
        selected.add(node.node_id)
        return True

    def _evaluate_selection(self, pos_sel: Sequence[int], neg_sel: Sequence[int]) -> Tuple[float, Dict[str, float], np.ndarray]:
        key = (tuple(sorted(pos_sel)), tuple(sorted(neg_sel)))
        if key in self.eval_cache:
            score, metrics, mask = self.eval_cache[key]
            return score, dict(metrics), mask.copy()

        pos_points = [[node.x, node.y] for node in self._selected_nodes(self.pos_nodes, pos_sel)]
        neg_points = [[node.x, node.y] for node in self._selected_nodes(self.neg_nodes, neg_sel)]
        mask = self.segmenter.segment(
            self.prompt_set.pseudo_rgb,
            pos_points,
            neg_points,
            self.prompt_set.change_score,
            self.prompt_set.valid_mask,
        ).astype(np.uint8)
        mask_bool = mask > 0
        valid = self.prompt_set.valid_mask
        dark_map = self.prompt_set.darkening_score
        log_map = self.prompt_set.log_ratio_score
        boundary_map = self.prompt_set.boundary_score
        inside_valid = mask_bool & valid
        outside_valid = (~mask_bool) & valid
        dark_inside = float(dark_map[inside_valid].mean()) if np.any(inside_valid) else 0.0
        dark_outside = float(dark_map[outside_valid].mean()) if np.any(outside_valid) else 0.0
        log_inside = float(log_map[inside_valid].mean()) if np.any(inside_valid) else 0.0
        mask_boundary = self._mask_boundary(mask_bool)
        boundary_alignment = float(boundary_map[mask_boundary & valid].mean()) if np.any(mask_boundary & valid) else 0.0
        area_ratio = float(inside_valid.sum() / max(valid.sum(), 1))
        area_penalty = self._area_penalty(area_ratio)
        metrics = {
            "darkening_support": dark_inside,
            "log_ratio_support": log_inside,
            "outside_contrast": float(np.clip(dark_inside - dark_outside, 0.0, 1.0)),
            "boundary_alignment": boundary_alignment,
            "component_quality": self._component_quality(mask),
            "shape_spread": self._spatial_dispersion(self._selected_nodes(self.pos_nodes, pos_sel)),
            "neg_ring_score": self._neg_ring_score(self._selected_nodes(self.pos_nodes, pos_sel), self._selected_nodes(self.neg_nodes, neg_sel)),
            "cross_sep": self._feature_separation(pos_sel, neg_sel),
            "redundancy": self._redundancy_penalty(pos_sel, neg_sel),
            "area_ratio": area_ratio,
            "area_penalty": area_penalty,
            "mean_pos_score": float(np.mean([self.pos_nodes[i].score for i in pos_sel])) if pos_sel else 0.0,
            "mean_neg_score": float(np.mean([self.neg_nodes[i].score for i in neg_sel])) if neg_sel else 0.0,
        }
        objective = 0.0
        for key_name, value in metrics.items():
            if key_name in self.objective_weights:
                objective += self.objective_weights[key_name] * value
        self.eval_cache[key] = (objective, dict(metrics), mask.copy())
        return objective, metrics, mask

    def _selected_nodes(self, table: Dict[int, CandidateNode], selected: Sequence[int]) -> List[CandidateNode]:
        return [table[idx] for idx in sorted(selected) if idx in table]

    def _worst_selected(self, table: Dict[int, CandidateNode], selected: Sequence[int]) -> CandidateNode | None:
        if not selected:
            return None
        return min((table[idx] for idx in selected), key=lambda node: node.score)

    def _best_reserve(self, sorted_nodes: Sequence[CandidateNode], selected: Sequence[int]) -> CandidateNode | None:
        selected = set(selected)
        for node in sorted_nodes:
            if node.node_id not in selected:
                return node
        return None

    def _swap_best(self, sorted_nodes: Sequence[CandidateNode], table: Dict[int, CandidateNode], selected: set[int]) -> bool:
        reserve = self._best_reserve(sorted_nodes, selected)
        worst = self._worst_selected(table, selected)
        if reserve is None or worst is None or reserve.node_id == worst.node_id or reserve.score <= worst.score:
            return False
        selected.remove(worst.node_id)
        selected.add(reserve.node_id)
        return True

    def _swap_diverse(self, sorted_nodes: Sequence[CandidateNode], table: Dict[int, CandidateNode], selected: set[int]) -> bool:
        reserve = self._diverse_reserve(sorted_nodes, selected, table)
        worst = self._worst_selected(table, selected)
        if reserve is None or worst is None or reserve.node_id == worst.node_id:
            return False
        selected.remove(worst.node_id)
        selected.add(reserve.node_id)
        return True

    def _diverse_reserve(self, sorted_nodes: Sequence[CandidateNode], selected: Sequence[int], table: Dict[int, CandidateNode]) -> CandidateNode | None:
        selected_nodes = self._selected_nodes(table, selected)
        if not selected_nodes:
            return self._best_reserve(sorted_nodes, selected)
        best_node = None
        best_score = -1.0
        selected_ids = set(selected)
        for node in sorted_nodes:
            if node.node_id in selected_ids:
                continue
            min_dist = min(_node_distance(node, other) for other in selected_nodes)
            score = 0.65 * node.score + 0.35 * min(1.0, min_dist / 96.0)
            if score > best_score:
                best_score = score
                best_node = node
        return best_node

    def _boundary_reserve(self, sorted_nodes: Sequence[CandidateNode], selected: Sequence[int]) -> CandidateNode | None:
        selected_ids = set(selected)
        best_node = None
        best_score = -1.0
        for node in sorted_nodes:
            if node.node_id in selected_ids:
                continue
            boundary_value = float(self.prompt_set.boundary_score[node.y, node.x])
            dark_value = float(self.prompt_set.darkening_score[node.y, node.x])
            score = 0.55 * node.score + 0.30 * boundary_value + 0.15 * dark_value
            if score > best_score:
                best_score = score
                best_node = node
        return best_node

    def _hard_negative_reserve(self, selected: Sequence[int]) -> CandidateNode | None:
        selected_ids = set(selected)
        pos_nodes = self._selected_nodes(self.pos_nodes, self.selected_pos)
        best_node = None
        best_score = -1.0
        for node in self.neg_sorted:
            if node.node_id in selected_ids:
                continue
            ring_score = self._node_ring_score(node, pos_nodes)
            stable = float(self.prompt_set.stable_score[node.y, node.x])
            score = 0.55 * node.score + 0.25 * ring_score + 0.20 * stable
            if score > best_score:
                best_score = score
                best_node = node
        return best_node

    def _spatial_dispersion(self, nodes: Sequence[CandidateNode]) -> float:
        if len(nodes) <= 1:
            return 0.0
        coords = np.asarray([[node.x, node.y] for node in nodes], dtype=np.float32)
        dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        return float(np.clip(dist[np.triu_indices(len(nodes), k=1)].mean() / 128.0, 0.0, 1.0))

    def _feature_separation(self, pos_sel: Sequence[int], neg_sel: Sequence[int]) -> float:
        pos_nodes = self._selected_nodes(self.pos_nodes, pos_sel)
        neg_nodes = self._selected_nodes(self.neg_nodes, neg_sel)
        if not pos_nodes or not neg_nodes:
            return 0.0
        pos_feat = np.asarray([node.feature for node in pos_nodes], dtype=np.float32)
        neg_feat = np.asarray([node.feature for node in neg_nodes], dtype=np.float32)
        dist = np.linalg.norm(pos_feat[:, None, :] - neg_feat[None, :, :], axis=-1).mean()
        return float(np.clip(dist / 5.0, 0.0, 1.0))

    def _redundancy_penalty(self, pos_sel: Sequence[int], neg_sel: Sequence[int]) -> float:
        penalty = 0.0
        for nodes in [self._selected_nodes(self.pos_nodes, pos_sel), self._selected_nodes(self.neg_nodes, neg_sel)]:
            if len(nodes) <= 1:
                continue
            coords = np.asarray([[node.x, node.y] for node in nodes], dtype=np.float32)
            dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
            penalty += float((dist[np.triu_indices(len(nodes), k=1)] < 10.0).mean())
        return float(np.clip(penalty / 2.0, 0.0, 1.0))

    def _component_quality(self, mask: np.ndarray) -> float:
        num_labels, _, stats = connected_components_with_stats(mask.astype(np.uint8))
        if num_labels <= 1:
            return 0.0
        areas = np.asarray([item["area"] for item in stats[1:]], dtype=np.float32)
        total = float(areas.sum())
        if total <= 0.0:
            return 0.0
        largest_ratio = float(areas.max() / total)
        tiny_ratio = float((areas < 20).mean())
        return float(np.clip(0.7 * largest_ratio + 0.3 * (1.0 - tiny_ratio), 0.0, 1.0))

    def _mask_boundary(self, mask: np.ndarray) -> np.ndarray:
        mask_u8 = mask.astype(np.uint8)
        return (binary_dilate(mask_u8, 3) > 0) & ~(binary_erode(mask_u8, 3) > 0)

    def _area_penalty(self, area_ratio: float) -> float:
        target = float(np.clip(self.prompt_set.area_prior, 0.01, 0.45))
        tolerance = max(0.03, 0.50 * target)
        deviation = max(abs(area_ratio - target) - tolerance, 0.0)
        return float(np.clip(deviation / max(target + tolerance, 1e-3), 0.0, 1.0))

    def _neg_ring_score(self, pos_nodes: Sequence[CandidateNode], neg_nodes: Sequence[CandidateNode]) -> float:
        if not pos_nodes or not neg_nodes:
            return 0.0
        scores = [self._node_ring_score(node, pos_nodes) for node in neg_nodes]
        return float(np.mean(scores)) if scores else 0.0

    def _node_ring_score(self, node: CandidateNode, pos_nodes: Sequence[CandidateNode]) -> float:
        if not pos_nodes:
            return 0.0
        min_dist = min(_node_distance(node, other) for other in pos_nodes)
        return float(np.exp(-((min_dist - 24.0) ** 2) / (2.0 * 14.0 * 14.0)))


def rule_greedy_optimize(env: PromptOptimizationEnv) -> OptimizationSummary:
    """不依赖标签与预训练 Q-table 的规则式贪心优化。"""
    env.reset()
    while env.steps < env.max_steps:
        best_action = None
        best_delta = 0.0
        for action in env.get_available_actions():
            delta, _, _ = env.evaluate_action_delta(action)
            if delta > best_delta:
                best_delta = delta
                best_action = action
        if best_action is None:
            break
        env.apply_action(best_action)
    return env.export_summary()


def _node_distance(a: CandidateNode, b: CandidateNode) -> float:
    return float(np.hypot(a.x - b.x, a.y - b.y))
