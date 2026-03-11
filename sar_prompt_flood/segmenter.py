"""GF3_Henan promptable 分割后端。"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from .ops import binary_close, binary_dilate, binary_open, draw_disk


class BasePromptSegmenter:
    """统一的 promptable 分割接口。"""

    def segment(
        self,
        pseudo_rgb: np.ndarray,
        pos_points: Sequence[Sequence[int]],
        neg_points: Sequence[Sequence[int]],
        change_score: np.ndarray,
        valid_mask: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError


class HeuristicFloodSegmenter(BasePromptSegmenter):
    """基于变化先验与 prompts 的启发式分割器。"""

    def segment(
        self,
        pseudo_rgb: np.ndarray,
        pos_points: Sequence[Sequence[int]],
        neg_points: Sequence[Sequence[int]],
        change_score: np.ndarray,
        valid_mask: np.ndarray,
    ) -> np.ndarray:
        h, w = change_score.shape
        if not pos_points:
            return np.zeros((h, w), dtype=np.uint8)

        seed = np.zeros((h, w), dtype=np.uint8)
        for x, y in pos_points:
            if 0 <= y < h and 0 <= x < w:
                seed = draw_disk(seed, (int(x), int(y)), 10, 1)
        seed = binary_dilate(seed, 7, iterations=1)
        q = float(np.quantile(change_score[valid_mask], 0.82)) if valid_mask.any() else 0.5
        q = max(q, 0.35)
        candidate = (change_score >= q).astype(np.uint8)
        candidate = binary_close(candidate, 5)
        candidate = binary_open(candidate, 3)
        mask = np.maximum(candidate, seed)
        for x, y in neg_points:
            if 0 <= y < h and 0 <= x < w:
                mask = draw_disk(mask, (int(x), int(y)), 8, 0)
        mask[~valid_mask] = 0
        return mask.astype(np.uint8)


@dataclass
class SamConfig:
    model_type: str
    checkpoint: str
    module_root: str


class SamPromptSegmenter(BasePromptSegmenter):
    """本地 Segment Anything 分割器。"""

    def __init__(self, cfg: SamConfig) -> None:
        module_root = Path(cfg.module_root).resolve()
        if str(module_root) not in sys.path:
            sys.path.insert(0, str(module_root))
        from segment_anything import SamPredictor, sam_model_registry  # pylint: disable=import-outside-toplevel

        ckpt = Path(cfg.checkpoint).expanduser().resolve()
        if not ckpt.exists():
            raise FileNotFoundError(f"SAM checkpoint not found: {ckpt}")
        self.predictor_cls = SamPredictor
        self.model = sam_model_registry[cfg.model_type](checkpoint=str(ckpt))

    def segment(
        self,
        pseudo_rgb: np.ndarray,
        pos_points: Sequence[Sequence[int]],
        neg_points: Sequence[Sequence[int]],
        change_score: np.ndarray,
        valid_mask: np.ndarray,
    ) -> np.ndarray:
        image = pseudo_rgb.astype(np.uint8)
        predictor = self.predictor_cls(self.model)
        predictor.set_image(image)
        coords = []
        labels = []
        for x, y in pos_points:
            coords.append([x, y])
            labels.append(1)
        for x, y in neg_points:
            coords.append([x, y])
            labels.append(0)
        if not coords:
            return np.zeros(image.shape[:2], dtype=np.uint8)
        masks, _, _ = predictor.predict(
            point_coords=np.asarray(coords, dtype=np.float32),
            point_labels=np.asarray(labels, dtype=np.int64),
            multimask_output=False,
        )
        mask = masks[0].astype(np.uint8)
        mask[~valid_mask] = 0
        return mask


def build_segmenters(cfg: dict) -> Tuple[HeuristicFloodSegmenter, Optional[SamPromptSegmenter]]:
    """始终构造启发式分割器，SAM 可选。"""
    heuristic = HeuristicFloodSegmenter()
    if cfg["segmenter"].get("backend", "sam") != "sam":
        return heuristic, None
    sam_cfg = SamConfig(
        model_type=cfg["segmenter"].get("sam_model_type", "vit_b"),
        checkpoint=cfg["segmenter"].get("sam_checkpoint", ""),
        module_root=cfg["segmenter"].get("sam_module_root", "PPO-main/segmenter"),
    )
    try:
        return heuristic, SamPromptSegmenter(sam_cfg)
    except Exception:
        return heuristic, None


def select_final_mask(sam_mask: np.ndarray | None, heuristic_mask: np.ndarray, change_score: np.ndarray, valid_mask: np.ndarray) -> Tuple[np.ndarray, str]:
    """按变化先验一致性选择最终 mask。"""
    if sam_mask is None:
        return heuristic_mask, "heuristic"
    sam_score = _score_mask(sam_mask, change_score, valid_mask)
    heu_score = _score_mask(heuristic_mask, change_score, valid_mask)
    if sam_score >= heu_score and sam_mask.sum() > 0:
        return sam_mask, "sam"
    return heuristic_mask, "heuristic"


def _score_mask(mask: np.ndarray, change_score: np.ndarray, valid_mask: np.ndarray) -> float:
    mask = (mask > 0) & valid_mask
    if not np.any(mask):
        return 0.0
    focus = float(change_score[mask].mean())
    area_ratio = float(mask.sum() / max(valid_mask.sum(), 1))
    area_penalty = 0.5 if area_ratio < 0.002 or area_ratio > 0.65 else 0.0
    return focus - area_penalty
