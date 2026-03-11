"""监督参考 prompt 生成。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch

from .prompts import (
    CandidateNode,
    PromptSet,
    _accept_node,
    _deduplicate_and_sort,
    _safe_quantile,
    _topk_coords,
    extract_node_feature,
)
from .reference_models import ReferencePromptProposalNet


@dataclass
class ProposalMaps:
    """参考引导热图输出。"""

    positive_prob: np.ndarray
    negative_prob: np.ndarray
    segmentation_prob: np.ndarray


class ReferencePromptCandidateGenerator:
    """把参考引导热图转换为 SAM prompt 集合。"""

    def __init__(
        self,
        model: ReferencePromptProposalNet,
        device: str = "cpu",
        patch_radius: int = 6,
        positive_quantile: float = 0.90,
        negative_quantile: float = 0.45,
        min_component_area: int = 20,
        min_point_distance: int = 16,
        max_positive_candidates: int = 18,
        max_negative_candidates: int = 18,
        initial_positive_points: int = 6,
        initial_negative_points: int = 6,
        low_confidence_max_score: float = 0.18,
        proposal_threshold: float = 0.55,
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

    @torch.no_grad()
    def predict_maps(self, query_sample: Dict, reference_sample: Dict) -> ProposalMaps:
        """预测 query 上的正负热图。"""
        self.model.eval()
        query_inputs = query_sample["features"].unsqueeze(0).to(self.device)
        ref_inputs = reference_sample["features"].unsqueeze(0).to(self.device)
        ref_gt = reference_sample["gt"].unsqueeze(0).to(self.device)
        valid_mask = reference_sample["valid_mask"].unsqueeze(0).to(self.device)
        output = self.model(query_inputs, ref_inputs, ref_gt, valid_mask=valid_mask)
        pos_prob = torch.sigmoid(output.positive_logit)[0, 0].cpu().numpy().astype(np.float32)
        neg_prob = torch.sigmoid(output.negative_logit)[0, 0].cpu().numpy().astype(np.float32)
        seg_prob = torch.sigmoid(output.segmentation_logit)[0, 0].cpu().numpy().astype(np.float32)
        return ProposalMaps(pos_prob, neg_prob, seg_prob)

    def generate(self, query_sample: Dict, reference_sample: Dict) -> PromptSet:
        """生成带参考引导的 prompt set。"""
        maps = self.predict_maps(query_sample, reference_sample)
        query_maps = self._build_query_maps(query_sample, maps)
        valid_mask = query_sample["valid_mask"].numpy().astype(bool)
        pos_nodes = self._generate_positive_nodes(query_maps, valid_mask)
        neg_nodes = self._generate_negative_nodes(query_maps, valid_mask, pos_nodes)
        low_conf = bool(float(query_maps["change_score"][valid_mask].max()) < self.low_confidence_max_score) if valid_mask.any() else True
        return PromptSet(
            positive_nodes=pos_nodes,
            negative_nodes=neg_nodes,
            initial_positive_ids=[n.node_id for n in pos_nodes[: self.initial_positive_points]],
            initial_negative_ids=[n.node_id for n in neg_nodes[: self.initial_negative_points]],
            change_score=query_maps["change_score"],
            stable_score=query_maps["stable_score"],
            boundary_score=query_maps["boundary_score"],
            pseudo_rgb=query_sample["pseudo_rgb"].numpy().astype(np.uint8),
            valid_mask=valid_mask,
            low_confidence=low_conf,
        )

    def _build_query_maps(self, query_sample: Dict, proposal_maps: ProposalMaps) -> Dict[str, np.ndarray]:
        pre = query_sample["pre"].numpy().astype(np.float32)
        post = query_sample["post"].numpy().astype(np.float32)
        diff = query_sample["diff"].numpy().astype(np.float32)
        change_score = query_sample["change_score"].numpy().astype(np.float32)
        stable_score = query_sample["stable_score"].numpy().astype(np.float32)
        boundary_score = query_sample["boundary_score"].numpy().astype(np.float32)
        log_ratio_like = query_sample["log_ratio_like"].numpy().astype(np.float32)
        pos_fused = np.clip(0.55 * proposal_maps.positive_prob + 0.30 * proposal_maps.segmentation_prob + 0.15 * change_score, 0.0, 1.0)
        neg_fused = np.clip(0.60 * proposal_maps.negative_prob + 0.40 * stable_score, 0.0, 1.0)
        return {
            "pre": pre,
            "post": post,
            "diff": diff,
            "change_score": np.clip(0.7 * change_score + 0.3 * pos_fused, 0.0, 1.0),
            "stable_score": stable_score,
            "boundary_score": boundary_score,
            "log_ratio_like": log_ratio_like,
            "positive_score": pos_fused,
            "negative_score": neg_fused,
        }

    def _generate_positive_nodes(self, maps: Dict[str, np.ndarray], valid_mask: np.ndarray) -> List[CandidateNode]:
        score = maps["positive_score"] * valid_mask.astype(np.float32)
        q = max(_safe_quantile(score[valid_mask], self.positive_quantile, fallback=self.proposal_threshold), self.proposal_threshold)
        binary = (score >= q).astype(np.uint8)
        binary[~valid_mask] = 0
        nodes: List[CandidateNode] = []
        next_id = 0
        num_labels = int(binary.max())
        if num_labels > 0:
            from .ops import connected_components_with_stats

            num_labels, labels, stats = connected_components_with_stats(binary)
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

    def _generate_negative_nodes(
        self,
        maps: Dict[str, np.ndarray],
        valid_mask: np.ndarray,
        pos_nodes: Sequence[CandidateNode],
    ) -> List[CandidateNode]:
        score = maps["negative_score"] * valid_mask.astype(np.float32)
        pos_score = maps["positive_score"]
        score[pos_score >= max(_safe_quantile(pos_score[valid_mask], self.positive_quantile, fallback=self.proposal_threshold), self.proposal_threshold)] = 0.0
        nodes: List[CandidateNode] = []
        next_id = 0
        for y, x in _topk_coords(score, k=self.max_negative_candidates * 8):
            if not valid_mask[y, x]:
                continue
            node = CandidateNode(next_id, "neg", int(y), int(x), float(score[y, x]), extract_node_feature(maps, y, x, self.patch_radius))
            if _accept_node(nodes, node, self.min_point_distance):
                nodes.append(node)
                next_id += 1
            if len(nodes) >= self.max_negative_candidates:
                break
        if not nodes and pos_nodes:
            for node in pos_nodes[: self.max_negative_candidates]:
                mirror = CandidateNode(next_id, "neg", node.y, max(node.x - 24, 0), 0.25, extract_node_feature(maps, node.y, max(node.x - 24, 0), self.patch_radius))
                nodes.append(mirror)
                next_id += 1
        return _deduplicate_and_sort(nodes, self.min_point_distance, self.max_negative_candidates)
