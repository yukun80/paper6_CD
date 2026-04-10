#!/usr/bin/env python3
"""使用 ChangeDINO 的 S1GFloods 权重对整景 SAR 切片做无标签拼接推理。"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.create_ChangeDINO import create_model  # noqa: E402
from model.blocks.dinov3_meta import get_dino_arch_spec, resolve_dino_arch, resolve_extract_ids  # noqa: E402
from option import (  # noqa: E402
    DEFAULT_BACKBONE_WEIGHT,
    Options,
    _validate_backbone_weight_path,
    normalize_contrast_pool_sizes,
    resolve_norm_stats,
)


PROB_NODATA = -1.0
BINARY_NODATA = 255
INFER_MODEL_CONFIG_FIELDS = {
    "backbone",
    "backbone_weight",
    "fpn_channels",
    "deform_groups",
    "gamma_mode",
    "beta_mode",
    "n_layers",
    "align_window",
    "align_points",
    "align_heads",
    "align_on_levels",
    "align_qkv_bias",
    "align_offset_groups",
    "p2_window_size",
    "micro_gate",
    "dino_collab_mode",
    "branch_consistency_weight",
    "consistency_warmup_epochs",
    "refiner",
    "contrast_pool_sizes",
    "contrast_pool_size",
    "topo_grid_size",
    "topo_hidden_dim",
    "topo_neighbor_k",
    "topo_neighbor_mode",
    "topo_long_offsets",
    "topo_n_hops",
    "topo_min_node_occ",
    "dino_arch",
    "dino_weight",
    "extract_ids",
}
SUPPORTED_BACKBONES = {"mobilenetv2", "efficientnet_b0"}


class TileDataset(Dataset):
    """读取预处理阶段生成的 PNG 切片与有效像素掩膜。"""

    def __init__(self, tiles_root: Path, mean: list[float], std: list[float]):
        self.tiles_root = tiles_root
        self.manifest_path = tiles_root / "tile_manifest.csv"
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Missing manifest: {self.manifest_path}")

        with self.manifest_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError(f"Manifest is empty: {self.manifest_path}")

        self.rows = sorted(rows, key=lambda row: int(row["tile_index"]))
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(tuple(mean), tuple(std))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        img_a = Image.open(self.tiles_root / row["a_png"]).convert("RGB")
        img_b = Image.open(self.tiles_root / row["b_png"]).convert("RGB")
        with rasterio.open(self.tiles_root / row["valid_mask"]) as ds_mask:
            valid_mask = ds_mask.read(1).astype(np.uint8, copy=False)

        return {
            "tile_id": row["tile_id"],
            "top": int(row["top"]),
            "left": int(row["left"]),
            "height": int(row["height"]),
            "width": int(row["width"]),
            "img1": self.normalize(self.to_tensor(img_a)),
            "img2": self.normalize(self.to_tensor(img_b)),
            "valid_mask": torch.from_numpy(valid_mask.copy()),
        }


def build_parser(
    description: str = "Infer ChangeDINO on SAR scene tiles",
    defaults: dict[str, object] | None = None,
) -> argparse.ArgumentParser:
    opt_builder = Options()
    opt_builder.init()
    parser = opt_builder.parser
    parser.set_defaults(
        name="S1GFloods-ChangeDINO",
        dataset="S1GFloods_CD_DINO",
        batch_size=8,
        num_workers=4,
        backbone="efficientnet_b0",
        backbone_weight=DEFAULT_BACKBONE_WEIGHT,
        stats_file="datasets/S1GFloods_CD_DINO/channel_stats_s1gfloods_train.json",
    )
    parser.description = description
    parser.add_argument("--tiles-root", type=Path, default=Path("datasets/SAR_Scene_CD_infer"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "ChangeDINO-main/checkpoints/S1GFloods-ChangeDINO-vitl16/"
            "S1GFloods-ChangeDINO-vitl16_efficientnet_b0_best.pth"
        ),
        help="trainval_s1gfloods.sh 训练得到的 checkpoint 路径。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ChangeDINO-main/outputs/sar_scene"),
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--skip-tiles",
        action="store_true",
        help="跳过切片级 PNG/TIF 保存，仅做整景拼接。",
    )
    if defaults:
        parser.set_defaults(**defaults)
    return parser


def parse_args(
    argv: Sequence[str] | None = None,
    defaults: dict[str, object] | None = None,
    description: str = "Infer ChangeDINO on SAR scene tiles",
) -> argparse.Namespace:
    return build_parser(description=description, defaults=defaults).parse_args(argv)


def collect_explicit_overrides(argv: Sequence[str] | None) -> set[str]:
    """记录用户显式传入的结构参数，保证 CLI 覆盖优先级高于 checkpoint 元数据。"""
    args = list(sys.argv[1:] if argv is None else argv)
    explicit: set[str] = set()
    for token in args:
        if not token.startswith("--"):
            continue
        key = token[2:].split("=", 1)[0]
        if key in INFER_MODEL_CONFIG_FIELDS:
            explicit.add(key)
    return explicit


def load_checkpoint_model_config(checkpoint_path: Path) -> dict[str, object] | None:
    """读取 checkpoint 中保存的模型结构元数据；旧 checkpoint 可能没有该字段。"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        return None
    meta = checkpoint.get("meta")
    if not isinstance(meta, dict):
        return None
    model_config = meta.get("model_config")
    return model_config if isinstance(model_config, dict) else None


def _extract_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "network" in checkpoint:
        state_dict = checkpoint["network"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError(f"Unsupported checkpoint payload type: {type(checkpoint)}")
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint state_dict type: {type(state_dict)}")
    return state_dict


def infer_local_dino_weight_for_arch(dino_arch: str) -> str | None:
    weight_dir = PROJECT_ROOT / "dinov3" / "weights"
    matches = sorted(weight_dir.glob(f"{dino_arch}*.pth"))
    if len(matches) == 1:
        return str(matches[0].resolve())
    return None


def count_transformer_blocks(state_dict: dict[str, torch.Tensor], prefix: str) -> int | None:
    pattern = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.")
    indices = set()
    for key in state_dict:
        match = pattern.match(key)
        if match:
            indices.add(int(match.group(1)))
    return (max(indices) + 1) if indices else None


def infer_checkpoint_model_config_from_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, object] | None:
    """兼容旧 checkpoint：从参数形状反推关键模型配置。"""
    cfg: dict[str, object] = {}

    def _infer_ocda_window_size(rel_pos_shape: int) -> int | None:
        for window_size in range(1, 65):
            rel_size = window_size + int(window_size * 0.5)
            if 2 * rel_size - 1 == rel_pos_shape:
                return window_size
        return None

    cls_key = "encoder.dino.model.cls_token"
    if cls_key in state_dict:
        embed_dim = int(state_dict[cls_key].shape[-1])
        arch_by_dim = {384: "dinov3_vits16", 768: "dinov3_vitb16", 1024: "dinov3_vitl16"}
        dino_arch = arch_by_dim.get(embed_dim)
        if dino_arch:
            cfg["dino_arch"] = dino_arch
            cfg["extract_ids"] = list(get_dino_arch_spec(dino_arch)["default_extract_ids"])
            inferred_weight = infer_local_dino_weight_for_arch(dino_arch)
            if inferred_weight:
                cfg["dino_weight"] = inferred_weight

    p5_head_key = "detector.p5_head.weight"
    if p5_head_key in state_dict:
        cfg["fpn_channels"] = int(state_dict[p5_head_key].shape[1])

    offset_key = "encoder.fpn.p2.conv_offset.weight"
    if offset_key in state_dict:
        offset_weight = state_dict[offset_key]
        cfg["deform_groups"] = int(offset_weight.shape[0] // 27)
        in_channels = int(offset_weight.shape[1])
        p4_in_channels = int(state_dict.get("encoder.fpn.p4.conv_offset.weight", torch.empty(0)).shape[1]) if "encoder.fpn.p4.conv_offset.weight" in state_dict else None
        if in_channels == 24 and p4_in_channels == 96:
            cfg["backbone"] = "mobilenetv2"
        elif in_channels == 24 and p4_in_channels == 112:
            cfg["backbone"] = "efficientnet_b0"
        elif in_channels == 80:
            raise ValueError(
                "This checkpoint was trained with convnextv2_nano, which is no longer supported in the ChangeDINO main path."
            )

    n_layers = []
    for prefix in ("detector.tb5", "detector.tb4", "detector.tb3", "detector.tb2"):
        depth = count_transformer_blocks(state_dict, prefix)
        if depth is None:
            n_layers = []
            break
        n_layers.append(depth)
    if n_layers:
        cfg["n_layers"] = n_layers
    cfg["contrast_pool_sizes"] = [5, 5, 5, 5]

    tb2_rel_key = "detector.tb2.0.spatial_attn.rel_pos_emb.rel_height"
    if tb2_rel_key in state_dict:
        inferred_p2_window = _infer_ocda_window_size(int(state_dict[tb2_rel_key].shape[0]))
        if inferred_p2_window is not None:
            cfg["p2_window_size"] = inferred_p2_window

    if any(key.startswith("refiner.topo_net") for key in state_dict):
        cfg["refiner"] = "topo"
    if any(key.startswith("refiner.contrast_refiner") for key in state_dict):
        cfg["refiner"] = "hybrid"
    if any(key.startswith("detector.micro_gate.tiny_head") for key in state_dict):
        cfg["micro_gate"] = True
    cfg["dino_collab_mode"] = "legacy"
    cfg["branch_consistency_weight"] = 0.0
    cfg["consistency_warmup_epochs"] = 0
    cfg["topo_neighbor_mode"] = "knn"
    cfg["topo_long_offsets"] = []
    if any(key.startswith("detector.p1_dino_gate") for key in state_dict) or any(
        key.startswith("detector.p3_dino_ctx") for key in state_dict
    ):
        cfg["dino_collab_mode"] = "multilevel_v2"
        cfg["branch_consistency_weight"] = 0.1

    if "backbone" not in cfg:
        cfg["backbone"] = "efficientnet_b0"
    if "gamma_mode" not in cfg:
        cfg["gamma_mode"] = "SE"
    if "beta_mode" not in cfg:
        cfg["beta_mode"] = "contextgatedconv"

    return cfg or None


def apply_checkpoint_model_config(
    opt: argparse.Namespace,
    checkpoint_model_config: dict[str, object] | None,
    explicit_overrides: set[str],
) -> tuple[argparse.Namespace, bool]:
    """用 checkpoint 里的结构参数补全当前推理配置，但不覆盖用户显式传参。"""
    if not checkpoint_model_config:
        return opt, False

    for field in INFER_MODEL_CONFIG_FIELDS:
        cli_name = field
        if cli_name in explicit_overrides:
            continue
        if field not in checkpoint_model_config:
            continue
        value = checkpoint_model_config[field]
        if field in {"n_layers", "extract_ids", "align_on_levels", "topo_long_offsets"} and value is not None:
            value = [int(v) for v in value]
        setattr(opt, field, value)
    return opt, True


def parse_and_prepare(
    argv: Sequence[str] | None = None,
    *,
    defaults: dict[str, object] | None = None,
    description: str = "Infer ChangeDINO on SAR scene tiles",
) -> argparse.Namespace:
    explicit_overrides = collect_explicit_overrides(argv)
    opt = parse_args(argv=argv, defaults=defaults, description=description)
    opt.tiles_root = Path(opt.tiles_root)
    opt.checkpoint = Path(opt.checkpoint)
    opt.output_dir = Path(opt.output_dir)

    if not opt.tiles_root.is_dir():
        raise FileNotFoundError(f"Tiles root not found: {opt.tiles_root}")
    if not opt.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {opt.checkpoint}")
    if not 0.0 <= opt.threshold <= 1.0:
        raise ValueError("--threshold must be within [0, 1]")
    if not opt.stats_file and not (opt.mean and opt.std):
        raise ValueError("Inference requires --stats_file or both --mean and --std")

    dino_weight = Path(opt.dino_weight)
    if not dino_weight.is_absolute():
        candidate = (PROJECT_ROOT / dino_weight).resolve()
        if candidate.is_file():
            opt.dino_weight = str(candidate)
    opt.tiles_root = opt.tiles_root.resolve()
    opt.checkpoint = opt.checkpoint.resolve()
    opt.output_dir = opt.output_dir.resolve()

    checkpoint_model_config = load_checkpoint_model_config(opt.checkpoint)
    inferred_checkpoint_model_config = None
    if checkpoint_model_config is None:
        inferred_checkpoint_model_config = infer_checkpoint_model_config_from_state_dict(
            _extract_state_dict(opt.checkpoint)
        )
        checkpoint_model_config = inferred_checkpoint_model_config
    opt, used_checkpoint_model_config = apply_checkpoint_model_config(
        opt, checkpoint_model_config, explicit_overrides
    )
    if (
        checkpoint_model_config
        and "dino_collab_mode" not in checkpoint_model_config
        and "dino_collab_mode" not in explicit_overrides
    ):
        opt.dino_collab_mode = "legacy"
    if (
        checkpoint_model_config
        and "branch_consistency_weight" not in checkpoint_model_config
        and "branch_consistency_weight" not in explicit_overrides
    ):
        opt.branch_consistency_weight = 0.0
    if (
        checkpoint_model_config
        and "consistency_warmup_epochs" not in checkpoint_model_config
        and "consistency_warmup_epochs" not in explicit_overrides
    ):
        opt.consistency_warmup_epochs = 0
    if (
        checkpoint_model_config
        and "topo_neighbor_mode" not in checkpoint_model_config
        and "topo_neighbor_mode" not in explicit_overrides
    ):
        opt.topo_neighbor_mode = "knn"
    if (
        checkpoint_model_config
        and "topo_long_offsets" not in checkpoint_model_config
        and "topo_long_offsets" not in explicit_overrides
    ):
        opt.topo_long_offsets = []
    opt.contrast_pool_sizes = normalize_contrast_pool_sizes(
        getattr(opt, "contrast_pool_sizes", None), getattr(opt, "contrast_pool_size", None)
    )
    if not hasattr(opt, "dino_collab_mode"):
        opt.dino_collab_mode = "multilevel_v2"
    if checkpoint_model_config is None:
        opt.dino_collab_mode = "legacy"
        opt.branch_consistency_weight = 0.0
        opt.consistency_warmup_epochs = 0
        opt.topo_neighbor_mode = "knn"
        opt.topo_long_offsets = []
    if opt.backbone not in SUPPORTED_BACKBONES:
        raise NotImplementedError(
            f"Unsupported backbone from CLI/checkpoint: {opt.backbone}. "
            "Only mobilenetv2 and efficientnet_b0 are supported."
        )

    if inferred_checkpoint_model_config is not None:
        print(
            "[WARN] Checkpoint does not contain model_config metadata. "
            "Inference inferred model structure from checkpoint tensor shapes."
        )
    elif not used_checkpoint_model_config:
        print(
            "[WARN] Checkpoint does not contain model_config metadata. "
            "Inference will rely on CLI/default backbone and DINO settings."
        )

    dino_weight = Path(opt.dino_weight)
    if not dino_weight.is_absolute():
        candidate = (PROJECT_ROOT / dino_weight).resolve()
        if candidate.is_file():
            opt.dino_weight = str(candidate)
    if opt.backbone_weight:
        _validate_backbone_weight_path(str(opt.backbone_weight))
        backbone_weight = Path(opt.backbone_weight)
        if not backbone_weight.is_absolute():
            candidate = (PROJECT_ROOT / backbone_weight).resolve()
            if candidate.is_file():
                opt.backbone_weight = str(candidate)
    if opt.stats_file:
        opt.stats_file = str(Path(opt.stats_file).resolve())

    str_ids = opt.gpu_ids.split(",")
    opt.gpu_ids = []
    for str_id in str_ids:
        gid = int(str_id)
        if gid >= 0:
            opt.gpu_ids.append(gid)
    if opt.gpu_ids and torch.cuda.is_available():
        torch.cuda.set_device(opt.gpu_ids[0])
        print(f"[INFO] Using CUDA device {opt.gpu_ids[0]}")
    else:
        opt.gpu_ids = []
        print("[WARN] CUDA is unavailable or gpu_ids is empty; inference will run on CPU")

    opt.phase = "test"
    opt.load_pretrain = False
    opt.dino_arch = resolve_dino_arch(opt.dino_arch, opt.dino_weight)
    opt.extract_ids = resolve_extract_ids(opt.dino_arch, opt.extract_ids)
    opt.mean, opt.std = resolve_norm_stats(opt)
    opt.output_dir.mkdir(parents=True, exist_ok=True)

    opt.mosaic_dir = opt.output_dir / "mosaic"
    opt.mosaic_dir.mkdir(parents=True, exist_ok=True)

    if opt.skip_tiles:
        opt.tile_png_dir = None
        opt.tile_tif_dir = None
    else:
        opt.tile_png_dir = opt.output_dir / "tile_png"
        opt.tile_tif_dir = opt.output_dir / "tile_tif"
        opt.tile_png_dir.mkdir(parents=True, exist_ok=True)
        opt.tile_tif_dir.mkdir(parents=True, exist_ok=True)

    print("------------ Options -------------")
    for key, value in sorted(vars(opt).items()):
        print(f"{key}: {value}")
    print("-------------- End ----------------")
    return opt


def load_model(opt: argparse.Namespace):
    """复用训练配置构建模型，并显式加载用户指定的 checkpoint。"""
    os.chdir(PROJECT_ROOT)
    model = create_model(opt)
    checkpoint = torch.load(opt.checkpoint, map_location=model.device, weights_only=True)
    state_dict = checkpoint["network"] if isinstance(checkpoint, dict) and "network" in checkpoint else checkpoint
    missing, unexpected = model.model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys when loading checkpoint: {len(missing)}")
    if unexpected:
        print(f"[WARN] Unexpected keys when loading checkpoint: {len(unexpected)}")
    model.eval()
    return model


def load_prepare_report(tiles_root: Path) -> dict[str, object]:
    report_path = tiles_root / "prepare_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Missing prepare report: {report_path}")
    return json.loads(report_path.read_text(encoding="utf-8"))


def resolve_source_path(path_str: str) -> Path:
    """兼容旧报告中的相对路径，避免导出阶段受 cwd 影响。"""
    path = Path(path_str)
    if path.is_absolute():
        return path

    candidates = [
        (PROJECT_ROOT.parent / path).resolve(),
        (Path.cwd() / path).resolve(),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return (PROJECT_ROOT.parent / path).resolve()


def build_blend_weight(size: int) -> np.ndarray:
    """使用中心权重窗减轻滑窗边界缝。"""
    if size <= 1:
        return np.ones((size, size), dtype=np.float32)

    axis = np.hanning(size).astype(np.float32)
    if float(axis.max()) <= 0.0:
        axis = np.ones((size,), dtype=np.float32)
    axis = np.clip(axis, 1e-3, None)
    return np.clip(np.outer(axis, axis).astype(np.float32), 1e-3, None)


def write_geotiff(
    source_path: Path,
    out_path: Path,
    arr: np.ndarray,
    dtype: str,
    nodata: float | int | None,
) -> None:
    with rasterio.open(source_path) as src:
        profile = src.profile.copy()
    profile.update(
        {
            "driver": "GTiff",
            "height": arr.shape[0],
            "width": arr.shape[1],
            "count": 1,
            "dtype": dtype,
            "compress": "LZW",
            "nodata": nodata,
        }
    )
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr, 1)


def write_preview_png(binary_map: np.ndarray, out_path: Path) -> None:
    """生成便于快速查看的 RGB 预览图。"""
    rgb = np.zeros((binary_map.shape[0], binary_map.shape[1], 3), dtype=np.uint8)
    valid_bg = binary_map == 0
    valid_fg = binary_map == 1
    rgb[valid_bg] = np.array([40, 40, 40], dtype=np.uint8)
    rgb[valid_fg] = np.array([255, 255, 255], dtype=np.uint8)
    Image.fromarray(rgb, mode="RGB").save(out_path)


def save_tile_png(
    prob: np.ndarray,
    valid_mask: np.ndarray,
    threshold: float,
    save_path: Path,
) -> None:
    """将切片概率图二值化后保存为灰度 PNG（L 模式，0/255）。"""
    binary = (prob >= threshold).astype(np.uint8)
    binary[valid_mask == 0] = 0
    pred_img = binary * 255
    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pred_img, mode="L").save(save_path)


def save_tile_tif(
    prob: np.ndarray,
    valid_mask: np.ndarray,
    threshold: float,
    save_path: Path,
) -> None:
    """将切片二值预测保存为 uint8 TIF（0/1，nodata=255）。"""
    binary = (prob >= threshold).astype(np.uint8)
    binary[valid_mask == 0] = BINARY_NODATA
    save_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": prob.shape[0],
        "width": prob.shape[1],
        "count": 1,
        "dtype": "uint8",
        "compress": "LZW",
        "nodata": BINARY_NODATA,
    }
    with rasterio.open(save_path, "w", **profile) as dst:
        dst.write(binary, 1)


def main(
    argv: Sequence[str] | None = None,
    *,
    defaults: dict[str, object] | None = None,
    description: str = "Infer ChangeDINO on SAR scene tiles",
) -> None:
    opt = parse_and_prepare(argv=argv, defaults=defaults, description=description)
    prepare_report = load_prepare_report(opt.tiles_root)
    dataset = TileDataset(opt.tiles_root, opt.mean, opt.std)
    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.num_workers),
        pin_memory=True,
    )

    source = prepare_report["source"]
    source_pre = resolve_source_path(str(source["pre_image"]))
    full_height = int(source["height"])
    full_width = int(source["width"])
    tile_size = int(prepare_report["params"]["tile_size"])

    accum_prob = np.zeros((full_height, full_width), dtype=np.float32)
    accum_weight = np.zeros((full_height, full_width), dtype=np.float32)
    base_weight = build_blend_weight(tile_size)

    model = load_model(opt)

    total_tiles_saved = 0
    with torch.no_grad():
        pbar = tqdm(dataloader, total=len(dataloader), ncols=100, desc="Infer tiles")
        for batch_idx, batch in enumerate(pbar, start=1):
            img1 = batch["img1"].to(model.device, non_blocking=torch.cuda.is_available())
            img2 = batch["img2"].to(model.device, non_blocking=torch.cuda.is_available())
            logits = model.inference(img1, img2)
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy().astype(np.float32, copy=False)
            valid_masks = batch["valid_mask"].numpy().astype(np.float32, copy=False)
            tile_ids = batch["tile_id"]
            tops = batch["top"].tolist()
            lefts = batch["left"].tolist()
            heights = batch["height"].tolist()
            widths = batch["width"].tolist()

            for idx in range(probs.shape[0]):
                top = int(tops[idx])
                left = int(lefts[idx])
                height = int(heights[idx])
                width = int(widths[idx])
                valid = valid_masks[idx, :height, :width]
                weight = base_weight[:height, :width] * valid
                if not np.any(weight > 0):
                    continue
                accum_prob[top : top + height, left : left + width] += probs[idx, :height, :width] * weight
                accum_weight[top : top + height, left : left + width] += weight

                if opt.tile_png_dir is not None:
                    tile_id = tile_ids[idx]
                    save_tile_png(
                        probs[idx, :height, :width],
                        valid[:height, :width],
                        opt.threshold,
                        opt.tile_png_dir / f"{tile_id}.png",
                    )
                    save_tile_tif(
                        probs[idx, :height, :width],
                        valid[:height, :width],
                        opt.threshold,
                        opt.tile_tif_dir / f"{tile_id}.tif",
                    )
                    total_tiles_saved += 1
            pbar.set_postfix({"batch": batch_idx, "tiles": min(batch_idx * opt.batch_size, len(dataset))})

    valid_output = accum_weight > 0
    prob_map = np.full((full_height, full_width), PROB_NODATA, dtype=np.float32)
    prob_map[valid_output] = accum_prob[valid_output] / np.maximum(accum_weight[valid_output], 1e-6)

    binary_map = np.full((full_height, full_width), BINARY_NODATA, dtype=np.uint8)
    binary_map[valid_output] = (prob_map[valid_output] >= float(opt.threshold)).astype(np.uint8)

    mosaic_dir = opt.mosaic_dir
    prob_path = mosaic_dir / "change_prob.tif"
    binary_tif_path = mosaic_dir / "change_binary.tif"
    binary_png_path = mosaic_dir / "change_binary.png"
    print("[INFO] Writing stitched outputs ...")
    write_geotiff(source_pre, prob_path, prob_map, "float32", PROB_NODATA)
    write_geotiff(source_pre, binary_tif_path, binary_map, "uint8", BINARY_NODATA)
    write_preview_png(binary_map, binary_png_path)

    report = {
        "tiles_root": str(opt.tiles_root),
        "checkpoint": str(opt.checkpoint),
        "stats_file": str(opt.stats_file) if opt.stats_file else "",
        "threshold": float(opt.threshold),
        "batch_size": int(opt.batch_size),
        "num_workers": int(opt.num_workers),
        "total_tiles": len(dataset),
        "tile_output": {
            "tile_png_dir": str(opt.tile_png_dir) if opt.tile_png_dir else "",
            "tile_tif_dir": str(opt.tile_tif_dir) if opt.tile_tif_dir else "",
            "total_saved": total_tiles_saved,
        },
        "output_files": {
            "change_prob_tif": str(prob_path),
            "change_binary_tif": str(binary_tif_path),
            "change_binary_png": str(binary_png_path),
        },
        "source_image": str(source_pre),
        "source_shape": [full_height, full_width],
        "model_config": {
            "backbone": opt.backbone,
            "fpn_channels": int(opt.fpn_channels),
            "deform_groups": int(opt.deform_groups),
            "gamma_mode": opt.gamma_mode,
            "beta_mode": opt.beta_mode,
            "n_layers": [int(v) for v in opt.n_layers],
            "align_window": int(opt.align_window),
            "align_points": int(opt.align_points),
            "align_heads": int(opt.align_heads),
            "align_on_levels": [int(v) for v in opt.align_on_levels],
            "align_qkv_bias": bool(opt.align_qkv_bias),
            "align_offset_groups": int(opt.align_offset_groups),
            "p2_window_size": int(getattr(opt, "p2_window_size", 8)),
            "micro_gate": bool(getattr(opt, "micro_gate", False)),
            "dino_collab_mode": getattr(opt, "dino_collab_mode", "multilevel_v2"),
            "branch_consistency_weight": float(
                getattr(opt, "branch_consistency_weight", 0.05)
            ),
            "consistency_warmup_epochs": int(
                getattr(opt, "consistency_warmup_epochs", 15)
            ),
            "refiner": getattr(opt, "refiner", "topo"),
            "contrast_pool_sizes": [
                int(v) for v in getattr(opt, "contrast_pool_sizes", [5, 5, 5, 5])
            ],
            "topo_grid_size": int(getattr(opt, "topo_grid_size", 16)),
            "topo_hidden_dim": int(getattr(opt, "topo_hidden_dim", 128)),
            "topo_neighbor_k": int(getattr(opt, "topo_neighbor_k", 12)),
            "topo_neighbor_mode": getattr(opt, "topo_neighbor_mode", "mixed"),
            "topo_long_offsets": [
                int(v) for v in getattr(opt, "topo_long_offsets", [2, 4])
            ],
            "topo_n_hops": int(getattr(opt, "topo_n_hops", 3)),
            "topo_min_node_occ": float(getattr(opt, "topo_min_node_occ", 0.25)),
            "dino_arch": opt.dino_arch,
            "dino_weight": str(opt.dino_weight),
            "extract_ids": [int(v) for v in opt.extract_ids],
        },
    }
    (opt.output_dir / "infer_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if total_tiles_saved > 0:
        print(f"[INFO] Saved tile predictions: {total_tiles_saved} (png={opt.tile_png_dir}, tif={opt.tile_tif_dir})")
    print(f"[DONE] tiles={len(dataset)} outputs={opt.output_dir}")


if __name__ == "__main__":
    main()
