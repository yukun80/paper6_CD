"""监督参考 prompt 生成。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch

from .ops import binary_dilate, connected_components_with_stats
from .prompts import CandidateNode, PromptSet, _accept_node, _deduplicate_and_sort, _safe_quantile, _topk_coords, extract_node_feature
from .reference_models import ReferencePromptProposalNet


@dataclass
class ProposalMaps:
    """参考引导热图输出。"""

    positive_prob: np.ndarray
    negative_prob: np.ndarray
    segmentation_prob: np.ndarray
    boundary_prob: np.ndarray


class ReferencePromptCandidateGenerator:
    """把参考引导热图转换为 SAM prompt 集合。"""

    def __init__(
        self,
        model: ReferencePromptProposalNet,
        device: str = "cpu",
        patch_radius: int = 6,
        positive_quantile: float = 0.88,
        negative_quantile: float = 0.40,
        min_component_area: int = 12,
        min_point_distance: int = 12,
        max_positive_candidates: int = 32,
        max_negative_candidates: int = 32,
        initial_positive_points: int = 10,
        initial_negative_points: int = 10,
        low_confidence_max_score: float = 0.16,
        proposal_threshold: float = 0.50,
        sampling_strategy: str = "multiscale_core_boundary",
    ) -> None:
        self.model = model
        self.device = device
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
        self.proposal_threshold = float(proposal_threshold)
        self.sampling_strategy = str(sampling_strategy)

    @torch.no_grad()
    def predict_maps(self, query_sample: Dict, reference_sample: Dict) -> ProposalMaps:
        """预测 query 上的正负热图。"""
        self.model.eval()
        query_inputs = query_sample["features"].unsqueeze(0).to(self.device)
        ref_inputs = reference_sample["features"].unsqueeze(0).to(self.device)
        ref_gt = reference_sample["gt"].unsqueeze(0).to(self.device)
        valid_mask = reference_sample["valid_mask"].unsqueeze(0).to(self.device)
        output = self.model(query_inputs, ref_inputs, ref_gt, valid_mask=valid_mask)
        return ProposalMaps(
            positive_prob=torch.sigmoid(output.positive_logit)[0, 0].cpu().numpy().astype(np.float32),
            negative_prob=torch.sigmoid(output.negative_logit)[0, 0].cpu().numpy().astype(np.float32),
            segmentation_prob=torch.sigmoid(output.segmentation_logit)[0, 0].cpu().numpy().astype(np.float32),
            boundary_prob=torch.sigmoid(output.boundary_logit)[0, 0].cpu().numpy().astype(np.float32),
        )

    def generate(self, query_sample: Dict, reference_sample: Dict) -> PromptSet:
        """生成带参考引导的 prompt set。"""
        maps = self.predict_maps(query_sample, reference_sample)
        query_maps = self._build_query_maps(query_sample, maps)
        valid_mask = query_sample["valid_mask"].numpy().astype(bool)
        pos_nodes = self._generate_positive_nodes(query_maps, valid_mask)
        neg_nodes = self._generate_negative_nodes(query_maps, valid_mask, pos_nodes)
        low_conf = bool(float(query_maps["positive_score"][valid_mask].max()) < self.low_confidence_max_score) if valid_mask.any() else True
        area_prior = float(((query_maps["positive_score"] >= max(0.55, self.proposal_threshold)) & valid_mask).sum() / max(valid_mask.sum(), 1))
        return PromptSet(
            positive_nodes=pos_nodes,
            negative_nodes=neg_nodes,
            initial_positive_ids=[n.node_id for n in pos_nodes[: self.initial_positive_points]],
            initial_negative_ids=[n.node_id for n in neg_nodes[: self.initial_negative_points]],
            change_score=query_maps["change_score"],
            darkening_score=query_maps["darkening_score"],
            log_ratio_score=query_maps["log_ratio_like"],
            stable_score=query_maps["stable_score"],
            boundary_score=query_maps["boundary_score"],
            pseudo_rgb=query_sample["pseudo_rgb"].numpy().astype(np.uint8),
            valid_mask=valid_mask,
            low_confidence=low_conf,
            area_prior=area_prior,
        )

    def _build_query_maps(self, query_sample: Dict, proposal_maps: ProposalMaps) -> Dict[str, np.ndarray]:
        pre = query_sample["pre"].numpy().astype(np.float32)
        post = query_sample["post"].numpy().astype(np.float32)
        diff = query_sample["diff"].numpy().astype(np.float32)
        darkening = query_sample["darkening_score"].numpy().astype(np.float32)
        brightening = query_sample["brightening_score"].numpy().astype(np.float32)
        change_score = query_sample["change_score"].numpy().astype(np.float32)
        stable_score = query_sample["stable_score"].numpy().astype(np.float32)
        boundary_score = query_sample["boundary_score"].numpy().astype(np.float32)
        log_ratio_like = query_sample["log_ratio_like"].numpy().astype(np.float32)
        local_contrast_score = query_sample["local_contrast_score"].numpy().astype(np.float32)
        pos_fused = np.clip(
            0.40 * proposal_maps.segmentation_prob
            + 0.25 * proposal_maps.positive_prob
            + 0.20 * darkening
            + 0.10 * log_ratio_like
            + 0.05 * local_contrast_score,
            0.0,
            1.0,
        )
        neg_fused = np.clip(
            0.40 * proposal_maps.negative_prob
            + 0.30 * stable_score
            + 0.20 * brightening
            + 0.10 * (1.0 - proposal_maps.segmentation_prob),
            0.0,
            1.0,
        )
        boundary_fused = np.clip(0.65 * proposal_maps.boundary_prob + 0.35 * boundary_score, 0.0, 1.0)
        return {
            "pre": pre,
            "post": post,
            "diff": diff,
            "darkening_score": darkening,
            "brightening_score": brightening,
            "change_score": np.clip(0.55 * change_score + 0.45 * pos_fused, 0.0, 1.0),
            "stable_score": stable_score,
            "boundary_score": boundary_fused,
            "log_ratio_like": log_ratio_like,
            "local_contrast_score": local_contrast_score,
            "positive_score": pos_fused,
            "negative_score": neg_fused,
        }

    def _generate_positive_nodes(self, maps: Dict[str, np.ndarray], valid_mask: np.ndarray) -> List[CandidateNode]:
        score = maps["positive_score"] * valid_mask.astype(np.float32)
        boundary_mix = np.clip(0.7 * score + 0.3 * maps["boundary_score"], 0.0, 1.0)
        small_mix = np.clip(0.7 * score + 0.3 * maps["local_contrast_score"], 0.0, 1.0)
        q = max(_safe_quantile(score[valid_mask], self.positive_quantile, fallback=self.proposal_threshold), self.proposal_threshold)
        binary = (score >= q).astype(np.uint8)
        binary[~valid_mask] = 0
        nodes: List[CandidateNode] = []
        next_id = 0
        num_labels, labels, stats = connected_components_with_stats(binary)
        positive_area = max(binary.sum(), 1)
        for comp_id in range(1, num_labels):
            area = int(stats[comp_id]["area"])
            if area < self.min_component_area:
                continue
            comp_mask = labels == comp_id
            budget = int(np.clip(round(self.max_positive_candidates * area / positive_area), 1, 6))
            next_id = self._append_from_score(nodes, maps, score * comp_mask, valid_mask, budget, next_id, "pos")
            next_id = self._append_from_score(nodes, maps, boundary_mix * comp_mask, valid_mask, max(1, budget // 2), next_id, "pos")
        next_id = self._append_from_score(nodes, maps, boundary_mix, valid_mask, self.max_positive_candidates, next_id, "pos")
        next_id = self._append_from_score(nodes, maps, small_mix, valid_mask, self.max_positive_candidates, next_id, "pos")
        return _deduplicate_and_sort(nodes, self.min_point_distance, self.max_positive_candidates)

    def _generate_negative_nodes(
        self,
        maps: Dict[str, np.ndarray],
        valid_mask: np.ndarray,
        pos_nodes: Sequence[CandidateNode],
    ) -> List[CandidateNode]:
        pos_score = maps["positive_score"]
        pos_binary = (pos_score >= max(_safe_quantile(pos_score[valid_mask], self.positive_quantile, fallback=self.proposal_threshold), self.proposal_threshold)).astype(np.uint8)
        pos_binary[~valid_mask] = 0
        near_ring = np.clip(binary_dilate(pos_binary, 9) - pos_binary, 0, 1).astype(bool)
        score = maps["negative_score"] * valid_mask.astype(np.float32)
        score[pos_binary > 0] = 0.0
        hard_neg = np.clip(0.7 * score + 0.3 * near_ring.astype(np.float32), 0.0, 1.0)
        nodes: List[CandidateNode] = []
        next_id = 0
        next_id = self._append_from_score(nodes, maps, hard_neg * near_ring.astype(np.float32), valid_mask, self.max_negative_candidates, next_id, "neg")
        next_id = self._append_from_score(nodes, maps, score, valid_mask, self.max_negative_candidates, next_id, "neg")
        if not nodes and pos_nodes:
            low_pos = np.clip(1.0 - pos_score, 0.0, 1.0)
            next_id = self._append_from_score(nodes, maps, low_pos * valid_mask.astype(np.float32), valid_mask, self.max_negative_candidates, next_id, "neg")
        return _deduplicate_and_sort(nodes, self.min_point_distance, self.max_negative_candidates)

    def _append_from_score(
        self,
        nodes: List[CandidateNode],
        maps: Dict[str, np.ndarray],
        score_map: np.ndarray,
        valid_mask: np.ndarray,
        budget: int,
        next_id: int,
        label: str,
    ) -> int:
        existing_count = len([node for node in nodes if node.label == label])
        for y, x in _topk_coords(score_map, k=max(budget * 6, budget)):
            if not valid_mask[y, x]:
                continue
            node = CandidateNode(
                next_id,
                label,
                int(y),
                int(x),
                float(score_map[y, x]),
                extract_node_feature(maps, y, x, self.patch_radius),
            )
            adaptive_distance = self._adaptive_min_distance(score_map, y, x)
            if _accept_node(nodes, node, adaptive_distance):
                nodes.append(node)
                next_id += 1
            if len([n for n in nodes if n.label == label]) - existing_count >= budget:
                break
        return next_id

    def _adaptive_min_distance(self, score_map: np.ndarray, y: int, x: int) -> int:
        window = score_map[max(0, y - 12) : y + 13, max(0, x - 12) : x + 13]
        local_mass = float((window > max(self.proposal_threshold, 0.45)).mean()) if window.size > 0 else 0.0
        if local_mass > 0.35:
            return max(8, self.min_point_distance - 4)
        if local_mass < 0.10:
            return self.min_point_distance + 4
        return self.min_point_distance
