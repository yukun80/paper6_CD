"""监督式 prompt 动作评分与优化。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .metrics import binary_dice, binary_iou
from .optimizer import ACTIONS, OptimizationSummary, PromptOptimizationEnv
from .prompts import CandidateNode

ACTION_TO_INDEX = {name: idx for idx, name in enumerate(ACTIONS)}


class PolicyActionScorer(nn.Module):
    """根据当前状态与候选动作预测增益。"""

    def __init__(self, feature_dim: int, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class ActionSample:
    """单个动作训练样本。"""

    feature: np.ndarray
    target: float
    action: str


FEATURE_KEYS: Tuple[str, ...] = (
    "overlap_high_change",
    "mask_focus",
    "coverage",
    "neg_coverage",
    "cross_sep",
    "redundancy",
    "fragments",
    "area_ratio",
    "area_penalty",
    "mean_pos_score",
    "mean_neg_score",
)


def collect_supervised_action_samples(env: PromptOptimizationEnv, gt: np.ndarray, ignore_index: int = 255) -> List[ActionSample]:
    """枚举当前状态可行动作，并用 GT 计算监督增益。"""
    current_iou = binary_iou(torch.from_numpy(env.current_mask), torch.from_numpy(gt), ignore_index=ignore_index)
    current_dice = binary_dice(torch.from_numpy(env.current_mask), torch.from_numpy(gt), ignore_index=ignore_index)
    samples: List[ActionSample] = []
    for action in env.get_available_actions():
        pos_sel = set(env.selected_pos)
        neg_sel = set(env.selected_neg)
        changed = env._apply_action_to_sets(action, pos_sel, neg_sel)  # pylint: disable=protected-access
        if not changed:
            continue
        _, metrics, mask = env._evaluate_selection(pos_sel, neg_sel)  # pylint: disable=protected-access
        next_iou = binary_iou(torch.from_numpy(mask), torch.from_numpy(gt), ignore_index=ignore_index)
        next_dice = binary_dice(torch.from_numpy(mask), torch.from_numpy(gt), ignore_index=ignore_index)
        target = float((next_iou - current_iou) + 0.5 * (next_dice - current_dice))
        feature = build_action_feature(env, action, metrics)
        samples.append(ActionSample(feature=feature, target=target, action=action))
    return samples


def build_action_feature(env: PromptOptimizationEnv, action: str, metrics: Dict[str, float] | None = None) -> np.ndarray:
    """构造动作评分器输入特征。"""
    metrics = metrics or env.current_metrics
    node = _resolve_action_node(env, action)
    one_hot = np.zeros(len(ACTIONS), dtype=np.float32)
    one_hot[ACTION_TO_INDEX[action]] = 1.0
    state_feats = np.asarray([float(metrics.get(k, 0.0)) for k in FEATURE_KEYS], dtype=np.float32)
    counts = np.asarray(
        [
            len(env.selected_pos) / max(env.max_positive_points, 1),
            len(env.selected_neg) / max(env.max_negative_points, 1),
            float(env.steps / max(env.max_steps, 1)),
        ],
        dtype=np.float32,
    )
    node_feats = _node_feature_vector(node, env)
    return np.concatenate([one_hot, counts, state_feats, node_feats], axis=0).astype(np.float32)


def policy_greedy_optimize(
    env: PromptOptimizationEnv,
    scorer: PolicyActionScorer,
    device: str = "cpu",
    stop_threshold: float = 0.0,
) -> OptimizationSummary:
    """用学习到的动作评分器进行贪心优化。"""
    env.reset()
    scorer.eval()
    while env.steps < env.max_steps:
        actions = env.get_available_actions()
        if not actions:
            break
        feats = np.stack([build_action_feature(env, action) for action in actions], axis=0)
        with torch.no_grad():
            scores = scorer(torch.from_numpy(feats).to(device)).detach().cpu().numpy()
        best_idx = int(np.argmax(scores))
        if float(scores[best_idx]) <= float(stop_threshold):
            break
        env.apply_action(actions[best_idx])
    return env.export_summary()


def infer_policy_feature_dim() -> int:
    return len(ACTIONS) + 3 + len(FEATURE_KEYS) + 16


def _resolve_action_node(env: PromptOptimizationEnv, action: str) -> CandidateNode | None:
    if action == "drop_pos_worst":
        return env._worst_selected(env.pos_nodes, env.selected_pos)  # pylint: disable=protected-access
    if action == "drop_neg_worst":
        return env._worst_selected(env.neg_nodes, env.selected_neg)  # pylint: disable=protected-access
    if action in {"restore_pos_best", "swap_pos_best", "swap_pos_diverse"}:
        if action == "swap_pos_diverse":
            return env._diverse_reserve(env.pos_sorted, env.selected_pos, env.pos_nodes)  # pylint: disable=protected-access
        return env._best_reserve(env.pos_sorted, env.selected_pos)  # pylint: disable=protected-access
    if action in {"restore_neg_best", "swap_neg_best"}:
        return env._best_reserve(env.neg_sorted, env.selected_neg)  # pylint: disable=protected-access
    if action == "swap_neg_boundary":
        return env._boundary_reserve(env.neg_sorted, env.selected_neg)  # pylint: disable=protected-access
    return None


def _node_feature_vector(node: CandidateNode | None, env: PromptOptimizationEnv) -> np.ndarray:
    if node is None:
        return np.zeros(16, dtype=np.float32)
    base = list(node.feature[:11])
    extras = [
        float(node.score),
        float(env.prompt_set.change_score[node.y, node.x]),
        float(env.prompt_set.stable_score[node.y, node.x]),
        float(env.prompt_set.boundary_score[node.y, node.x]),
        float(node.label == "pos"),
    ]
    return np.asarray(base + extras, dtype=np.float32)
