"""GF3 无标注 prompt 优化环境与规则优化器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .ops import connected_components_with_stats
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
)


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
        max_positive_points: int = 10,
        min_negative_points: int = 2,
        max_negative_points: int = 10,
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
            actions.extend(["restore_pos_best", "swap_pos_best", "swap_pos_diverse"])
        if len(self.selected_neg) < self.max_negative_points and len(self.selected_neg) < len(self.neg_nodes):
            actions.extend(["restore_neg_best", "swap_neg_best", "swap_neg_boundary"])
        if len(self.selected_pos) >= self.max_positive_points:
            actions.extend(["swap_pos_best", "swap_pos_diverse"])
        if len(self.selected_neg) >= self.max_negative_points:
            actions.extend(["swap_neg_best", "swap_neg_boundary"])
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
        return OptimizationSummary(pos_points=pos_points, neg_points=neg_points, mask=self.current_mask.copy(), metrics=dict(self.current_metrics), action_history=list(self.action_history))

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
        return False

    def _evaluate_selection(self, pos_sel: Sequence[int], neg_sel: Sequence[int]) -> Tuple[float, Dict[str, float], np.ndarray]:
        key = (tuple(sorted(pos_sel)), tuple(sorted(neg_sel)))
        if key in self.eval_cache:
            score, metrics, mask = self.eval_cache[key]
            return score, dict(metrics), mask.copy()

        pos_points = [[node.x, node.y] for node in self._selected_nodes(self.pos_nodes, pos_sel)]
        neg_points = [[node.x, node.y] for node in self._selected_nodes(self.neg_nodes, neg_sel)]
        mask = self.segmenter.segment(self.prompt_set.pseudo_rgb, pos_points, neg_points, self.prompt_set.change_score, self.prompt_set.valid_mask).astype(np.uint8)

        valid = self.prompt_set.valid_mask
        high_thr = float(np.quantile(self.prompt_set.change_score[valid], 0.90)) if valid.any() else 1.0
        high_change = (self.prompt_set.change_score >= max(high_thr, 0.4)) & valid
        mask_bool = mask > 0
        overlap = float((mask_bool & high_change).sum() / max(high_change.sum(), 1))
        focus = float(self.prompt_set.change_score[mask_bool & valid].mean()) if np.any(mask_bool & valid) else 0.0
        area_ratio = float(mask_bool.sum() / max(valid.sum(), 1))
        area_penalty = 1.0 if area_ratio < 0.002 or area_ratio > 0.65 else 0.0
        metrics = {
            "overlap_high_change": overlap,
            "mask_focus": focus,
            "coverage": self._spatial_dispersion(self._selected_nodes(self.pos_nodes, pos_sel)),
            "neg_coverage": self._spatial_dispersion(self._selected_nodes(self.neg_nodes, neg_sel)),
            "cross_sep": self._feature_separation(pos_sel, neg_sel),
            "redundancy": self._redundancy_penalty(pos_sel, neg_sel),
            "fragments": self._fragment_penalty(mask),
            "area_ratio": area_ratio,
            "area_penalty": area_penalty,
            "mean_pos_score": float(np.mean([self.pos_nodes[i].score for i in pos_sel])) if pos_sel else 0.0,
            "mean_neg_score": float(np.mean([self.neg_nodes[i].score for i in neg_sel])) if neg_sel else 0.0,
        }
        objective = (
            0.35 * metrics["overlap_high_change"]
            + 0.25 * metrics["mask_focus"]
            + 0.15 * metrics["cross_sep"]
            + 0.10 * metrics["coverage"]
            + 0.05 * metrics["neg_coverage"]
            + 0.05 * metrics["mean_pos_score"]
            + 0.05 * metrics["mean_neg_score"]
            - 0.15 * metrics["redundancy"]
            - 0.08 * metrics["fragments"]
            - 0.10 * metrics["area_penalty"]
        )
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
            score = 0.7 * node.score + 0.3 * min(1.0, min_dist / 64.0)
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
            score = 0.7 * node.score + 0.3 * float(self.prompt_set.boundary_score[node.y, node.x])
            if score > best_score:
                best_score = score
                best_node = node
        return best_node

    def _spatial_dispersion(self, nodes: Sequence[CandidateNode]) -> float:
        if len(nodes) <= 1:
            return 0.0
        coords = np.asarray([[node.x, node.y] for node in nodes], dtype=np.float32)
        dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        return float(np.clip(dist[np.triu_indices(len(nodes), k=1)].mean() / 96.0, 0.0, 1.0))

    def _feature_separation(self, pos_sel: Sequence[int], neg_sel: Sequence[int]) -> float:
        pos_nodes = self._selected_nodes(self.pos_nodes, pos_sel)
        neg_nodes = self._selected_nodes(self.neg_nodes, neg_sel)
        if not pos_nodes or not neg_nodes:
            return 0.0
        pos_feat = np.asarray([node.feature for node in pos_nodes], dtype=np.float32)
        neg_feat = np.asarray([node.feature for node in neg_nodes], dtype=np.float32)
        dist = np.linalg.norm(pos_feat[:, None, :] - neg_feat[None, :, :], axis=-1).mean()
        return float(np.clip(dist / 4.0, 0.0, 1.0))

    def _redundancy_penalty(self, pos_sel: Sequence[int], neg_sel: Sequence[int]) -> float:
        penalty = 0.0
        for nodes in [self._selected_nodes(self.pos_nodes, pos_sel), self._selected_nodes(self.neg_nodes, neg_sel)]:
            if len(nodes) <= 1:
                continue
            coords = np.asarray([[node.x, node.y] for node in nodes], dtype=np.float32)
            dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
            penalty += float((dist[np.triu_indices(len(nodes), k=1)] < 12.0).mean())
        return float(np.clip(penalty / 2.0, 0.0, 1.0))

    def _fragment_penalty(self, mask: np.ndarray) -> float:
        num_labels, _, stats = connected_components_with_stats(mask.astype(np.uint8))
        if num_labels <= 1:
            return 0.0
        areas = np.asarray([s["area"] for s in stats[1:]], dtype=np.float32)
        tiny_ratio = float((areas < 24).mean()) if areas.size > 0 else 0.0
        fragment_penalty = min(1.0, (num_labels - 1) / 12.0)
        return float(np.clip(0.5 * fragment_penalty + 0.5 * tiny_ratio, 0.0, 1.0))


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
