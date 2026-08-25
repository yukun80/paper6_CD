from pathlib import Path

import timm
import torch
import torch.nn as nn

DEFAULT_BACKBONE_NAME = "efficientnet_b2"
DEFAULT_BACKBONE_WEIGHT = "pretrained/efficientnet_b2_ra-bcdf34b7.pth"
_ALLOWED_CLASSIFICATION_HEAD_PREFIXES = ("conv_head.", "bn2.", "classifier.")


class TimmFeatureBackbone(nn.Module):
    """统一 timm features_only 骨干的输出接口。"""

    def __init__(self, model: nn.Module, channels: list[int], reductions: list[int] | None = None):
        super().__init__()
        self.model = model
        self.channels = channels
        self.reductions = reductions or []

    def forward(self, x):
        features = self.model(x)
        if isinstance(features, tuple):
            features = list(features)
        return features


def _load_local_backbone_weights(backbone: nn.Module, weight_path: str) -> None:
    """严格加载本地 B2 特征权重，只允许丢弃分类头。"""
    weight_file = Path(weight_path).expanduser()
    if not weight_file.is_file():
        raise FileNotFoundError(f"backbone weight not found: {weight_file}")

    checkpoint = torch.load(weight_file, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict):
        state_dict = (
            checkpoint.get("state_dict")
            or checkpoint.get("model")
            or checkpoint.get("network")
            or checkpoint
        )
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError(f"unsupported backbone checkpoint type: {type(state_dict)}")

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        clean_key = str(key)
        for prefix in ("module.", "model.", "backbone."):
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix) :]
        cleaned_state_dict[clean_key] = value

    try:
        missing, unexpected = backbone.load_state_dict(cleaned_state_dict, strict=False)
    except RuntimeError as error:
        raise RuntimeError(
            "EfficientNet-B2 feature checkpoint is incompatible; "
            "B0 and other backbone weights are unsupported"
        ) from error
    invalid_unexpected = [
        key
        for key in unexpected
        if not key.startswith(_ALLOWED_CLASSIFICATION_HEAD_PREFIXES)
    ]
    if missing or invalid_unexpected:
        raise RuntimeError(
            "EfficientNet-B2 feature checkpoint is incompatible: "
            f"missing={missing}, invalid_unexpected={invalid_unexpected}"
        )

    print(f"loaded local backbone weights for {DEFAULT_BACKBONE_NAME}: {weight_file}")
    print(f"backbone load summary | missing: {len(missing)} | unexpected: {len(unexpected)}")
    if unexpected:
        print(f"unexpected classification-head keys: {unexpected}")


def _build_timm_feature_backbone(backbone_weight: str) -> TimmFeatureBackbone:
    timm_backbone = timm.create_model(
        DEFAULT_BACKBONE_NAME,
        pretrained=False,
        features_only=True,
    )
    feature_info = timm_backbone.feature_info
    backbone = TimmFeatureBackbone(
        timm_backbone,
        channels=list(feature_info.channels()),
        reductions=list(feature_info.reduction()),
    )
    _load_local_backbone_weights(backbone.model, backbone_weight)
    return backbone


def build_feature_backbone(
    backbone_weight: str | None = DEFAULT_BACKBONE_WEIGHT,
) -> TimmFeatureBackbone:
    """构建 HA-CQI 唯一支持的 EfficientNet-B2 feature backbone。"""
    if not backbone_weight:
        raise ValueError("EfficientNet-B2 requires a local PyTorch --backbone_weight")
    return _build_timm_feature_backbone(backbone_weight)
