"""GF3_Henan 纯 SAM promptable 分割后端。"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


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


@dataclass
class SamConfig:
    """SAM 初始化参数。"""

    model_type: str
    checkpoint: str
    module_root: str


class SamPromptSegmenter(BasePromptSegmenter):
    """本地 Segment Anything 分割器。

    为了适配优化循环，这里复用单个 predictor 实例，只在每次调用时更新图像。
    """

    def __init__(self, cfg: SamConfig) -> None:
        module_root = Path(cfg.module_root).resolve()
        if str(module_root) not in sys.path:
            sys.path.insert(0, str(module_root))
        from segment_anything import SamPredictor, sam_model_registry  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel

        ckpt = Path(cfg.checkpoint).expanduser().resolve()
        if not ckpt.exists():
            raise FileNotFoundError(f"SAM checkpoint not found: {ckpt}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = sam_model_registry[cfg.model_type](checkpoint=str(ckpt))
        self.model.to(device=self.device)
        self.model.eval()
        self.predictor = SamPredictor(self.model)

    def segment(
        self,
        pseudo_rgb: np.ndarray,
        pos_points: Sequence[Sequence[int]],
        neg_points: Sequence[Sequence[int]],
        change_score: np.ndarray,
        valid_mask: np.ndarray,
    ) -> np.ndarray:
        image = pseudo_rgb.astype(np.uint8)
        self.predictor.set_image(image)
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
        masks, _, _ = self.predictor.predict(
            point_coords=np.asarray(coords, dtype=np.float32),
            point_labels=np.asarray(labels, dtype=np.int64),
            multimask_output=False,
        )
        mask = masks[0].astype(np.uint8)
        mask[~valid_mask] = 0
        return mask


def build_segmenter(cfg: dict) -> SamPromptSegmenter:
    """按配置构造单一 SAM 分割器。"""
    if cfg["segmenter"].get("backend", "sam") != "sam":
        raise ValueError("GF3_Henan pipeline only supports backend='sam'")
    sam_cfg = SamConfig(
        model_type=cfg["segmenter"].get("sam_model_type", "vit_b"),
        checkpoint=cfg["segmenter"].get("sam_checkpoint", ""),
        module_root=cfg["segmenter"].get("sam_module_root", "PPO-main/segmenter"),
    )
    return SamPromptSegmenter(sam_cfg)
