from pathlib import Path

import timm
import torch
import torch.nn as nn

from .mobilenetv2 import mobilenet_v2


DEFAULT_BACKBONE_WEIGHT = "pretrained/efficientnet_b0_ra-3dd342df.pth"


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


def _load_local_backbone_weights(backbone: nn.Module, weight_path: str, backbone_name: str) -> None:
    """优先加载本地骨干预训练权重，避免训练入口隐式联网。"""
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

    missing, unexpected = backbone.load_state_dict(cleaned_state_dict, strict=False)
    print(f"loaded local backbone weights for {backbone_name}: {weight_file}")
    print(f"backbone load summary | missing: {len(missing)} | unexpected: {len(unexpected)}")
    if missing:
        print(f"missing keys sample: {missing[:5]}")
    if unexpected:
        print(f"unexpected keys sample: {unexpected[:5]}")


def _build_timm_feature_backbone(
    model_name: str, backbone_weight: str | None = None
) -> TimmFeatureBackbone:
    timm_backbone = timm.create_model(model_name, pretrained=False, features_only=True)
    feature_info = timm_backbone.feature_info
    backbone = TimmFeatureBackbone(
        timm_backbone,
        channels=list(feature_info.channels()),
        reductions=list(feature_info.reduction()),
    )
    if backbone_weight:
        _load_local_backbone_weights(backbone.model, backbone_weight, model_name)
    return backbone


def build_feature_backbone(
    backbone_name: str, backbone_weight: str | None = DEFAULT_BACKBONE_WEIGHT
) -> nn.Module:
    """构建 HA-CQI 使用的 CNN feature backbone。"""
    if backbone_name == "mobilenetv2":
        backbone = mobilenet_v2(pretrained=True, progress=True)
        backbone.channels = [16, 24, 32, 96, 320]
        backbone.reductions = [2, 4, 8, 16, 32]
        return backbone

    if backbone_name == "efficientnet_b0":
        if not backbone_weight:
            raise ValueError("efficientnet_b0 requires a local PyTorch --backbone_weight")
        return _build_timm_feature_backbone("efficientnet_b0", backbone_weight=backbone_weight)

    raise NotImplementedError(
        f"BACKBONE [{backbone_name}] is not implemented. Supported: mobilenetv2, efficientnet_b0"
    )
