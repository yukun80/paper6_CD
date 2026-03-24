from __future__ import annotations

from pathlib import Path


DINO_ARCH_SPECS = {
    "dinov3_vits16": {
        "embed_dim": 384,
        "num_layers": 12,
        "patch_size": 16,
        "default_extract_ids": [2, 5, 8, 11],
    },
    "dinov3_vitb16": {
        "embed_dim": 768,
        "num_layers": 12,
        "patch_size": 16,
        "default_extract_ids": [2, 5, 8, 11],
    },
    "dinov3_vitl16": {
        "embed_dim": 1024,
        "num_layers": 24,
        "patch_size": 16,
        "default_extract_ids": [5, 11, 17, 23],
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


def resolve_extract_ids(dino_arch: str, extract_ids: list[int] | None) -> list[int]:
    spec = get_dino_arch_spec(dino_arch)
    resolved = list(spec["default_extract_ids"]) if extract_ids is None else [int(v) for v in extract_ids]
    if not resolved:
        raise ValueError("extract_ids must not be empty")
    invalid = [layer_id for layer_id in resolved if layer_id < 0 or layer_id >= int(spec["num_layers"])]
    if invalid:
        raise ValueError(
            f"extract_ids {invalid} exceed layer range for {dino_arch} "
            f"(valid: 0..{int(spec['num_layers']) - 1})"
        )
    return resolved
