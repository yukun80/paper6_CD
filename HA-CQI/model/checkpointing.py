"""HA-CQI checkpoint v2 的严格加载、配置解析与原子保存。"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any

import torch
from torch import nn


CHECKPOINT_FORMAT_VERSION = 2
FIXED_MODEL_CONFIG = {
    "backbone": "efficientnet_b2",
    "dino_input_norm": "imagenet",
    "decoder": "oscd_v1",
    "decoder_channels": 128,
    "ssm_state_dim": 1,
    "ssm_directions": 4,
    "context_levels": [3, 4, 5],
    "detail_levels": [2, 1],
}

# 仅这些字段会改变推理网络的构建方式。Loss、optimizer 和验证参数不属于模型结构。
INFERENCE_MODEL_CONFIG_FIELDS = frozenset(
    {
        "backbone",
        "backbone_weight",
        "fpn_channels",
        "deform_groups",
        "gamma_mode",
        "beta_mode",
        "disable_soft_alignment",
        "align_window",
        "align_points",
        "align_heads",
        "align_on_levels",
        "align_qkv_bias",
        "align_offset_groups",
        "num_change_queries",
        "cqi_heads",
        "decoder",
        "decoder_channels",
        "ssm_state_dim",
        "ssm_directions",
        "context_levels",
        "detail_levels",
        "dino_arch",
        "dino_weight",
        "dino_fusion_layers",
        "dino_input_norm",
        "input_mean",
        "input_std",
    }
)
LIST_MODEL_CONFIG_FIELDS = frozenset(
    {
        "align_on_levels",
        "context_levels",
        "detail_levels",
        "dino_fusion_layers",
        "input_mean",
        "input_std",
    }
)
MODEL_CONFIG_OPTION_FIELDS = {
    "input_mean": "mean",
    "input_std": "std",
}


def validate_checkpoint_v2(payload: dict[str, Any]) -> dict[str, Any]:
    """验证 checkpoint v2 顶层契约并返回 metadata。"""
    meta = payload.get("meta")
    if (
        not isinstance(meta, dict)
        or int(meta.get("format_version", 0)) != CHECKPOINT_FORMAT_VERSION
    ):
        raise ValueError(
            "HA-CQI only supports checkpoint format v2; legacy/raw checkpoints are unsupported"
        )
    state = payload.get("network")
    if (
        not isinstance(state, dict)
        or not state
        or not all(
            isinstance(key, str) and torch.is_tensor(value)
            for key, value in state.items()
        )
    ):
        raise ValueError("Checkpoint v2 must contain a non-empty tensor network state_dict")
    return meta


def load_checkpoint_payload(
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported checkpoint payload type: {type(payload)}")
    validate_checkpoint_v2(payload)
    return payload


def extract_network_state(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    validate_checkpoint_v2(payload)
    return payload["network"]


def checkpoint_model_config(payload: dict[str, Any]) -> dict[str, Any]:
    meta = validate_checkpoint_v2(payload)
    raw_model_config = meta.get("model_config")
    if not isinstance(raw_model_config, dict) or not raw_model_config:
        raise ValueError("Checkpoint v2 inference requires non-empty meta.model_config")
    model_config = dict(raw_model_config)
    # 20260823/24 的 B2-OSCD v2 checkpoint 记录了抽取四层、adapter 再丢首层。
    # 这里仅对这个精确且可证明等价的契约显式归一化，不接受其他旧格式或猜测。
    legacy_extract_ids = model_config.get("extract_ids")
    legacy_effective_layers = {
        ("dinov3_vits16", (2, 5, 8, 11)): [5, 8, 11],
        ("dinov3_vitb16", (2, 5, 8, 11)): [5, 8, 11],
        ("dinov3_vitl16", (5, 11, 17, 23)): [11, 17, 23],
    }.get(
        (
            model_config.get("dino_arch"),
            tuple(legacy_extract_ids) if isinstance(legacy_extract_ids, list) else (),
        )
    )
    if (
        "dino_fusion_layers" not in model_config
        and legacy_effective_layers is not None
        and model_config.get("decoder") == "oscd_v1"
        and model_config.get("backbone") == "efficientnet_b2"
    ):
        model_config["dino_fusion_layers"] = legacy_effective_layers
        model_config.pop("extract_ids", None)
        warnings.warn(
            "Normalized an early B2-OSCD v2 checkpoint from "
            f"extract_ids={legacy_extract_ids} to its actual effective fusion layers "
            f"{legacy_effective_layers}.",
            UserWarning,
            stacklevel=2,
        )
    if model_config.get("architecture") != "HA-CQI":
        raise ValueError("Checkpoint v2 meta.model_config.architecture must be 'HA-CQI'")
    if model_config.get("decoder") != "oscd_v1":
        raise ValueError(
            "HA-CQI OSCD-only decoder contract mismatch: "
            f"checkpoint decoder={model_config.get('decoder')!r}, required='oscd_v1'"
        )
    missing = sorted(INFERENCE_MODEL_CONFIG_FIELDS.difference(model_config))
    if missing:
        raise ValueError(
            "Checkpoint v2 meta.model_config lacks required inference fields: "
            + ", ".join(missing)
        )
    mismatches = [
        f"{field}: checkpoint={model_config.get(field)!r}, required={expected!r}"
        for field, expected in FIXED_MODEL_CONFIG.items()
        if model_config.get(field) != expected
    ]
    if mismatches:
        raise ValueError(
            "HA-CQI B2/OSCD-only model contract mismatch: " + "; ".join(mismatches)
        )
    fusion_layers = model_config.get("dino_fusion_layers")
    try:
        from .modules.dino_meta import resolve_dino_fusion_layers

        resolved_fusion_layers = resolve_dino_fusion_layers(
            str(model_config.get("dino_arch")),
            fusion_layers if isinstance(fusion_layers, list) else None,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Checkpoint model contract has invalid dino_fusion_layers: "
            f"{fusion_layers!r}"
        ) from error
    if fusion_layers != resolved_fusion_layers:
        raise ValueError(
            "Checkpoint model contract requires three ordered unique "
            f"dino_fusion_layers, got {fusion_layers!r}"
        )
    return model_config


def checkpoint_data_config(payload: dict[str, Any]) -> dict[str, Any] | None:
    meta = validate_checkpoint_v2(payload)
    data_config = meta.get("data_config")
    return data_config if isinstance(data_config, dict) else None


def resolve_inference_threshold(
    payload: dict[str, Any],
    explicit_threshold: float | None = None,
) -> tuple[float, str]:
    """按“显式 CLI → checkpoint selection”解析推理阈值，不提供隐式回退。"""
    meta = validate_checkpoint_v2(payload)
    if explicit_threshold is not None:
        threshold = float(explicit_threshold)
        source = "explicit_cli"
    else:
        selection = meta.get("selection")
        if not isinstance(selection, dict) or selection.get("threshold") is None:
            raise ValueError(
                "Inference requires --threshold or checkpoint v2 meta.selection.threshold"
            )
        threshold = float(selection["threshold"])
        source = "checkpoint_selection"
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Inference threshold must be within [0, 1], got {threshold}")
    return threshold, source


def apply_checkpoint_model_config(
    opt: Any,
    model_config: dict[str, Any],
    explicit_overrides: set[str],
) -> Any:
    """用 v2 结构配置补全推理参数，不覆盖用户显式 CLI。"""
    for field in INFERENCE_MODEL_CONFIG_FIELDS:
        option_field = MODEL_CONFIG_OPTION_FIELDS.get(field, field)
        if option_field in explicit_overrides or field not in model_config:
            continue
        value = model_config[field]
        if field in LIST_MODEL_CONFIG_FIELDS and value is not None:
            converter = float if field in {"input_mean", "input_std"} else int
            value = [converter(item) for item in value]
        setattr(opt, option_field, value)
    return opt


def cpu_state_dict(network: nn.Module) -> dict[str, torch.Tensor]:
    """复制 CPU state_dict，不改变在线网络设备。"""
    return {key: value.detach().cpu() for key, value in network.state_dict().items()}


def atomic_torch_save(payload: dict[str, Any], path_like: str | Path) -> Path:
    path = Path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path


def atomic_json_save(payload: dict[str, Any], path_like: str | Path) -> Path:
    path = Path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)
    return path
