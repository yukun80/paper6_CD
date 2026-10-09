from __future__ import annotations

from pathlib import Path


DINO_ARCH_SPECS = {
    "dinov3_vits16": {
        "embed_dim": 384,
        "num_layers": 12,
        "patch_size": 16,
        "default_fusion_layers": [5, 8, 11],
    },
    "dinov3_vitb16": {
        "embed_dim": 768,
        "num_layers": 12,
        "patch_size": 16,
        "default_fusion_layers": [5, 8, 11],
    },
    "dinov3_vitl16": {
        "embed_dim": 1024,
        "num_layers": 24,
        "patch_size": 16,
        "default_fusion_layers": [11, 17, 23],
    },
}

DINO_ARCH_CHOICES = ["auto", *DINO_ARCH_SPECS.keys()]


def infer_dino_arch_from_weights(weights_path: str | Path) -> str | None:
    """从本地权重文件名推断 DINO 架构，避免 CLI 必须重复写两遍。"""
    stem = Path(weights_path).stem.lower()
    for arch in DINO_ARCH_SPECS:
        if arch in stem:
            return arch
    return None


def resolve_dino_arch(dino_arch: str, weights_path: str | Path) -> str:
    if dino_arch != "auto":
        if dino_arch not in DINO_ARCH_SPECS:
            raise ValueError(f"Unsupported --dino_arch: {dino_arch}")
        return dino_arch

    inferred = infer_dino_arch_from_weights(weights_path)
    if inferred is None:
        raise ValueError(
            f"Cannot infer DINO arch from weights path: {weights_path}. "
            f"Please pass --dino_arch explicitly from {list(DINO_ARCH_SPECS.keys())}."
        )
    return inferred


def get_dino_arch_spec(dino_arch: str) -> dict[str, object]:
    if dino_arch not in DINO_ARCH_SPECS:
        raise ValueError(f"Unsupported DINO arch: {dino_arch}")
    return DINO_ARCH_SPECS[dino_arch]


def resolve_dino_fusion_layers(
    dino_arch: str,
    fusion_layers: list[int] | None,
) -> list[int]:
    """解析真正进入 P3-P5 融合的三个 DINO 层，禁止静默丢层。"""
    spec = get_dino_arch_spec(dino_arch)
    resolved = (
        list(spec["default_fusion_layers"])
        if fusion_layers is None
        else [int(v) for v in fusion_layers]
    )
    if len(resolved) != 3:
        raise ValueError(
            "dino_fusion_layers must contain exactly three layers for P3/P4/P5, "
            f"got {resolved}"
        )
    if len(set(resolved)) != len(resolved) or resolved != sorted(resolved):
        raise ValueError(
            "dino_fusion_layers must be unique and ordered from shallow to deep, "
            f"got {resolved}"
        )
    invalid = [layer_id for layer_id in resolved if layer_id < 0 or layer_id >= int(spec["num_layers"])]
    if invalid:
        raise ValueError(
            f"dino_fusion_layers {invalid} exceed layer range for {dino_arch} "
            f"(valid: 0..{int(spec['num_layers']) - 1})"
        )
    return resolved
