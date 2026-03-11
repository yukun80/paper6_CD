"""SAR prompt flood supervised reference package."""

from .feature_utils import robust_unit, safe_log_ratio
from .metrics import binary_dice, binary_iou
from .optimizer import PromptOptimizationEnv, rule_greedy_optimize
from .reference_config import load_reference_config
from .reference_data import GF3TargetTileDataset, UrbanSARReferenceTileDataset, ensure_reference_splits
from .segmenter import build_segmenter
from .supervised_prompts import ReferencePromptCandidateGenerator

__all__ = [
    "robust_unit",
    "safe_log_ratio",
    "binary_dice",
    "binary_iou",
    "PromptOptimizationEnv",
    "rule_greedy_optimize",
    "load_reference_config",
    "UrbanSARReferenceTileDataset",
    "GF3TargetTileDataset",
    "ensure_reference_splits",
    "build_segmenter",
    "ReferencePromptCandidateGenerator",
]
