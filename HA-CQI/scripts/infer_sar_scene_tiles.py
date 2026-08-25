#!/usr/bin/env python3
"""
使用 HA-CQI checkpoint 对整景 SAR 切片做无标签拼接推理。



"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio
import torch
from PIL import Image
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.engine import build_hacqi_engine  # noqa: E402
from model.backbones import DEFAULT_BACKBONE_WEIGHT  # noqa: E402
from model.checkpointing import (  # noqa: E402
    apply_checkpoint_model_config,
    checkpoint_data_config,
    checkpoint_model_config,
    extract_network_state,
    load_checkpoint_payload,
    resolve_inference_threshold,
)
from model.modules.dino_meta import (  # noqa: E402
    resolve_dino_arch,
    resolve_extract_ids,
)
from option import (  # noqa: E402
    Options,
    _validate_backbone_weight_path,
    resolve_norm_stats,
    validate_stats_provenance,
)


PROB_NODATA = -1.0
BINARY_NODATA = 255


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


def _path_candidates(path: Path) -> list[Path]:
    if path.is_absolute():
        return [path]
    return [
        (Path.cwd() / path).resolve(),
        (PROJECT_ROOT / path).resolve(),
        (REPO_ROOT / path).resolve(),
    ]


def resolve_existing_path(path_like: str | Path, *, expect_file: bool | None = None) -> Path:
    """兼容从仓库根目录或 HA-CQI 目录启动脚本的输入路径解析。"""
    path = Path(path_like).expanduser()
    candidates = _path_candidates(path)
    for candidate in candidates:
        if expect_file is True and candidate.is_file():
            return candidate
        if expect_file is False and candidate.is_dir():
            return candidate
        if expect_file is None and candidate.exists():
            return candidate
    return candidates[0]


def resolve_output_path(path_like: str | Path) -> Path:
    """输出路径默认按当前启动目录解析；脚本默认值使用 HA-CQI 绝对路径。"""
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def resolve_project_output_path(path_like: str | Path) -> Path:
    """HA-CQI 内部输出路径按项目目录解析，避免写到仓库根目录。"""
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def build_parser(
    description: str = "Infer HA-CQI on SAR scene tiles",
    defaults: dict[str, object] | None = None,
) -> argparse.ArgumentParser:
    opt_builder = Options()
    opt_builder.init()
    parser = opt_builder.parser
    parser.set_defaults(
        name="HA-CQI",
        dataset="S1GFloods_CD_DINO_BG_75_25",
        batch_size=8,
        num_workers=4,
        backbone_weight=DEFAULT_BACKBONE_WEIGHT,
        stats_file="datasets/S1GFloods_CD_DINO_BG_75_25/channel_stats_s1gfloods_train.json",
    )
    parser.description = description
    parser.add_argument("--tiles-root", type=Path, default=Path("datasets/SAR_Scene_CD_infer"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "sar_scene",
    )
    parser.add_argument(
        "--skip-tiles",
        action="store_true",
        help="跳过切片级 PNG/TIF 保存，仅做整景拼接。",
    )
    parser.add_argument(
        "--disable_blob_filter",
        action="store_true",
        help="关闭基于 overlap 一致性的整景伪斑过滤，保留原始拼接结果。",
    )
    parser.add_argument("--blob_filter_area_thresh", type=int, default=1024)
    parser.add_argument("--blob_filter_fill_thresh", type=float, default=0.45)
    parser.add_argument("--blob_filter_std_thresh", type=float, default=0.20)
    parser.add_argument("--blob_filter_vote_thresh", type=float, default=0.55)
    if defaults:
        parser.set_defaults(**defaults)
    return parser


def parse_args(
    argv: Sequence[str] | None = None,
    defaults: dict[str, object] | None = None,
    description: str = "Infer HA-CQI on SAR scene tiles",
) -> argparse.Namespace:
    return build_parser(description=description, defaults=defaults).parse_args(argv)


def collect_explicit_arguments(argv: Sequence[str] | None) -> set[str]:
    """收集全部显式 CLI 字段，用于 threshold/stats 的确定性优先级。"""
    args = list(sys.argv[1:] if argv is None else argv)
    return {
        token[2:].split("=", 1)[0].replace("-", "_")
        for token in args
        if token.startswith("--")
    }


def _resolve_model_weight_path(path_like: str | Path) -> str:
    path = Path(path_like).expanduser()
    for candidate in _path_candidates(path):
        if candidate.is_file():
            return str(candidate)
    return str(path)


def parse_and_prepare(
    argv: Sequence[str] | None = None,
    *,
    defaults: dict[str, object] | None = None,
    description: str = "Infer HA-CQI on SAR scene tiles",
) -> tuple[argparse.Namespace, dict[str, object]]:
    explicit_arguments = collect_explicit_arguments(argv)
    opt = parse_args(argv=argv, defaults=defaults, description=description)

    opt.tiles_root = resolve_existing_path(opt.tiles_root, expect_file=False)
    if not opt.checkpoint:
        raise ValueError("--checkpoint is required for scene inference")
    opt.checkpoint = resolve_existing_path(opt.checkpoint, expect_file=True)
    opt.output_dir = resolve_output_path(opt.output_dir)
    if not opt.tiles_root.is_dir():
        raise FileNotFoundError(f"Tiles root not found: {opt.tiles_root}")
    if not opt.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {opt.checkpoint}")

    checkpoint_payload = load_checkpoint_payload(opt.checkpoint, map_location="cpu")
    model_config = checkpoint_model_config(checkpoint_payload)
    opt = apply_checkpoint_model_config(
        opt,
        model_config,
        explicit_arguments,
    )

    data_config = checkpoint_data_config(checkpoint_payload) or {}
    if "stats_file" not in explicit_arguments and data_config.get("stats_file"):
        opt.stats_file = data_config["stats_file"]
    if "dataset" not in explicit_arguments and data_config.get("dataset"):
        opt.dataset = data_config["dataset"]
    if "dataroot" not in explicit_arguments and data_config.get("dataroot"):
        opt.dataroot = data_config["dataroot"]

    opt.threshold, opt.threshold_source = resolve_inference_threshold(
        checkpoint_payload,
        explicit_threshold=opt.threshold,
    )
    if opt.blob_filter_area_thresh < 1:
        raise ValueError("--blob_filter_area_thresh must be >= 1")
    for name in ("blob_filter_fill_thresh", "blob_filter_std_thresh", "blob_filter_vote_thresh"):
        value = float(getattr(opt, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name} must be within [0, 1]")
    if not opt.stats_file and not (opt.mean and opt.std):
        raise ValueError("Inference requires --stats_file or both --mean and --std")

    if str(opt.dataset_mode) == "auto":
        opt.dataset_mode = "sar" if str(opt.dataset).startswith("S1GFloods") else "default"
    if opt.disable_soft_alignment:
        opt.align_on_levels = []
    else:
        opt.align_on_levels = sorted({int(v) for v in getattr(opt, "align_on_levels", [1, 2, 3])})
        invalid_align_levels = [v for v in opt.align_on_levels if v not in {1, 2, 3}]
        if invalid_align_levels:
            raise ValueError(f"--align_on_levels only supports P1/P2/P3, got {invalid_align_levels}")
    opt.dataroot = str(resolve_existing_path(opt.dataroot, expect_file=False))
    if opt.stats_file:
        opt.stats_file = str(resolve_existing_path(opt.stats_file, expect_file=True))
    opt.checkpoint_dir = str(resolve_project_output_path(opt.checkpoint_dir))
    opt.dino_weight = _resolve_model_weight_path(opt.dino_weight)
    if opt.backbone_weight:
        opt.backbone_weight = _resolve_model_weight_path(opt.backbone_weight)
        _validate_backbone_weight_path(str(opt.backbone_weight))

    str_ids = str(opt.gpu_ids).split(",")
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
    opt.dino_arch = resolve_dino_arch(opt.dino_arch, opt.dino_weight)
    opt.extract_ids = resolve_extract_ids(opt.dino_arch, opt.extract_ids)
    opt.mean, opt.std = resolve_norm_stats(opt)
    validate_stats_provenance(opt)
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
    return opt, checkpoint_payload


def load_model(opt: argparse.Namespace, checkpoint_payload: dict[str, object]):
    """构建 HA-CQI 并显式加载用户指定的 checkpoint。"""
    model = build_hacqi_engine(opt)
    model.model.load_state_dict(extract_network_state(checkpoint_payload), strict=True)
    model.eval()
    return model


def load_prepare_report(tiles_root: Path) -> dict[str, object]:
    report_path = tiles_root / "prepare_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Missing prepare report: {report_path}")
    return json.loads(report_path.read_text(encoding="utf-8"))


def resolve_source_path(path_str: str) -> Path:
    """兼容 prepare_report 中的相对源影像路径。"""
    return resolve_existing_path(path_str, expect_file=True)


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
    rgb[binary_map == 0] = np.array([40, 40, 40], dtype=np.uint8)
    rgb[binary_map == 1] = np.array([255, 255, 255], dtype=np.uint8)
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
    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(binary * 255, mode="L").save(save_path)


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


def filter_unstable_blobs(
    binary_map: np.ndarray,
    valid_output: np.ndarray,
    prob_std_map: np.ndarray,
    vote_ratio_map: np.ndarray,
    *,
    area_thresh: int,
    fill_thresh: float,
    std_thresh: float,
    vote_thresh: float,
) -> tuple[np.ndarray, dict[str, int]]:
    """移除跨重叠 tile 不一致、但在单块内过于实心的大伪斑。"""
    filtered = binary_map.copy()
    fg_mask = (binary_map == 1) & valid_output
    if not np.any(fg_mask):
        return filtered, {"removed_components": 0, "removed_pixels": 0}

    structure = np.ones((3, 3), dtype=np.int8)
    cc_map, n_components = ndimage.label(fg_mask, structure=structure)
    removed_components = 0
    removed_pixels = 0

    for component_id in range(1, n_components + 1):
        mask = cc_map == component_id
        area = int(mask.sum())
        if area < area_thresh:
            continue

        ys, xs = np.where(mask)
        height = int(ys.max() - ys.min() + 1)
        width = int(xs.max() - xs.min() + 1)
        fill_ratio = area / max(height * width, 1)
        if fill_ratio < fill_thresh:
            continue

        mean_std = float(prob_std_map[mask].mean())
        mean_vote = float(vote_ratio_map[mask].mean())
        if mean_std < std_thresh or mean_vote >= vote_thresh:
            continue

        filtered[mask] = 0
        removed_components += 1
        removed_pixels += area

    return filtered, {
        "removed_components": removed_components,
        "removed_pixels": removed_pixels,
    }


def build_model_report(opt: argparse.Namespace) -> dict[str, object]:
    """记录 HA-CQI 推理复现实验所需的模型配置。"""
    return {
        "architecture": "HA-CQI",
        "backbone": opt.backbone,
        "fpn_channels": int(opt.fpn_channels),
        "deform_groups": int(opt.deform_groups),
        "gamma_mode": opt.gamma_mode,
        "beta_mode": opt.beta_mode,
        "disable_soft_alignment": bool(getattr(opt, "disable_soft_alignment", False)),
        "align_window": int(opt.align_window),
        "align_points": int(opt.align_points),
        "align_heads": int(opt.align_heads),
        "align_on_levels": [int(v) for v in opt.align_on_levels],
        "align_qkv_bias": bool(opt.align_qkv_bias),
        "align_offset_groups": int(opt.align_offset_groups),
        "num_change_queries": int(getattr(opt, "num_change_queries", 16)),
        "cqi_heads": int(getattr(opt, "cqi_heads", 4)),
        "decoder": "oscd_v1",
        "decoder_channels": 128,
        "ssm_state_dim": 1,
        "ssm_directions": 4,
        "context_levels": [3, 4, 5],
        "detail_levels": [2, 1],
        "dino_arch": opt.dino_arch,
        "dino_weight": str(opt.dino_weight),
        "extract_ids": [int(v) for v in opt.extract_ids],
        "dino_input_norm": "imagenet",
    }


def main(
    argv: Sequence[str] | None = None,
    *,
    defaults: dict[str, object] | None = None,
    description: str = "Infer HA-CQI on SAR scene tiles",
) -> None:
    opt, checkpoint_payload = parse_and_prepare(
        argv=argv,
        defaults=defaults,
        description=description,
    )
    prepare_report = load_prepare_report(opt.tiles_root)
    dataset = TileDataset(opt.tiles_root, opt.mean, opt.std)
    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.num_workers),
        pin_memory=torch.cuda.is_available(),
    )

    source = prepare_report["source"]
    source_pre = resolve_source_path(str(source["pre_image"]))
    full_height = int(source["height"])
    full_width = int(source["width"])
    tile_size = int(prepare_report["params"]["tile_size"])

    accum_prob = np.zeros((full_height, full_width), dtype=np.float32)
    accum_prob_sq = np.zeros((full_height, full_width), dtype=np.float32)
    accum_pos_weight = np.zeros((full_height, full_width), dtype=np.float32)
    accum_weight = np.zeros((full_height, full_width), dtype=np.float32)
    base_weight = build_blend_weight(tile_size)

    model = load_model(opt, checkpoint_payload)

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
                tile_prob = probs[idx, :height, :width]
                accum_prob[top : top + height, left : left + width] += tile_prob * weight
                accum_prob_sq[top : top + height, left : left + width] += np.square(tile_prob) * weight
                accum_pos_weight[top : top + height, left : left + width] += (
                    (tile_prob >= float(opt.threshold)).astype(np.float32) * weight
                )
                accum_weight[top : top + height, left : left + width] += weight

                if opt.tile_png_dir is not None:
                    tile_id = tile_ids[idx]
                    save_tile_png(
                        tile_prob,
                        valid,
                        opt.threshold,
                        opt.tile_png_dir / f"{tile_id}.png",
                    )
                    save_tile_tif(
                        tile_prob,
                        valid,
                        opt.threshold,
                        opt.tile_tif_dir / f"{tile_id}.tif",
                    )
                    total_tiles_saved += 1
            pbar.set_postfix({"batch": batch_idx, "tiles": min(batch_idx * opt.batch_size, len(dataset))})

    valid_output = accum_weight > 0
    prob_map = np.full((full_height, full_width), PROB_NODATA, dtype=np.float32)
    prob_map[valid_output] = accum_prob[valid_output] / np.maximum(accum_weight[valid_output], 1e-6)
    prob_var_map = np.zeros((full_height, full_width), dtype=np.float32)
    prob_var_map[valid_output] = (
        accum_prob_sq[valid_output] / np.maximum(accum_weight[valid_output], 1e-6)
        - np.square(prob_map[valid_output])
    )
    prob_std_map = np.zeros((full_height, full_width), dtype=np.float32)
    prob_std_map[valid_output] = np.sqrt(np.clip(prob_var_map[valid_output], 0.0, None))
    vote_ratio_map = np.zeros((full_height, full_width), dtype=np.float32)
    vote_ratio_map[valid_output] = (
        accum_pos_weight[valid_output] / np.maximum(accum_weight[valid_output], 1e-6)
    )

    raw_binary_map = np.full((full_height, full_width), BINARY_NODATA, dtype=np.uint8)
    raw_binary_map[valid_output] = (prob_map[valid_output] >= float(opt.threshold)).astype(np.uint8)

    if opt.disable_blob_filter:
        binary_map = raw_binary_map.copy()
        blob_filter_stats = {"removed_components": 0, "removed_pixels": 0}
    else:
        binary_map, blob_filter_stats = filter_unstable_blobs(
            raw_binary_map,
            valid_output,
            prob_std_map,
            vote_ratio_map,
            area_thresh=int(opt.blob_filter_area_thresh),
            fill_thresh=float(opt.blob_filter_fill_thresh),
            std_thresh=float(opt.blob_filter_std_thresh),
            vote_thresh=float(opt.blob_filter_vote_thresh),
        )

    mosaic_dir = opt.mosaic_dir
    prob_path = mosaic_dir / "change_prob.tif"
    raw_binary_tif_path = mosaic_dir / "change_binary_raw.tif"
    raw_binary_png_path = mosaic_dir / "change_binary_raw.png"
    binary_tif_path = mosaic_dir / "change_binary.tif"
    binary_png_path = mosaic_dir / "change_binary.png"
    print("[INFO] Writing stitched outputs ...")
    write_geotiff(source_pre, prob_path, prob_map, "float32", PROB_NODATA)
    write_geotiff(source_pre, raw_binary_tif_path, raw_binary_map, "uint8", BINARY_NODATA)
    write_preview_png(raw_binary_map, raw_binary_png_path)
    write_geotiff(source_pre, binary_tif_path, binary_map, "uint8", BINARY_NODATA)
    write_preview_png(binary_map, binary_png_path)

    report = {
        "tiles_root": str(opt.tiles_root),
        "checkpoint": str(opt.checkpoint),
        "stats_file": str(opt.stats_file) if opt.stats_file else "",
        "threshold": float(opt.threshold),
        "threshold_source": str(opt.threshold_source),
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
            "change_binary_raw_tif": str(raw_binary_tif_path),
            "change_binary_raw_png": str(raw_binary_png_path),
            "change_binary_tif": str(binary_tif_path),
            "change_binary_png": str(binary_png_path),
        },
        "blob_filter": {
            "enabled": not bool(opt.disable_blob_filter),
            "area_thresh": int(opt.blob_filter_area_thresh),
            "fill_thresh": float(opt.blob_filter_fill_thresh),
            "std_thresh": float(opt.blob_filter_std_thresh),
            "vote_thresh": float(opt.blob_filter_vote_thresh),
            **blob_filter_stats,
        },
        "source_image": str(source_pre),
        "source_shape": [full_height, full_width],
        "model_config": build_model_report(opt),
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
