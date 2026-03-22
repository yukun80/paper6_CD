#!/usr/bin/env python3
"""使用 ChangeDINO 的 S1GFloods 权重对整景 SAR 切片做无标签拼接推理。"""

from __future__ import annotations

import argparse
import csv
import json
import os
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
from model.blocks.dinov3_meta import resolve_dino_arch, resolve_extract_ids  # noqa: E402
from option import Options, resolve_norm_stats  # noqa: E402


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
        backbone="mobilenetv2",
        stats_file="datasets/S1GFloods_CD_DINO/channel_stats_s1gfloods_train.json",
    )
    parser.description = description
    parser.add_argument("--tiles-root", type=Path, default=Path("datasets/SAR_Scene_CD_infer"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("ChangeDINO-main/checkpoints/S1GFloods-ChangeDINO/S1GFloods-ChangeDINO_mobilenetv2_best.pth"),
        help="trainval_s1gfloods.sh 训练得到的 checkpoint 路径。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ChangeDINO-main/outputs/sar_scene"),
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    if defaults:
        parser.set_defaults(**defaults)
    return parser


def parse_args(
    argv: Sequence[str] | None = None,
    defaults: dict[str, object] | None = None,
    description: str = "Infer ChangeDINO on SAR scene tiles",
) -> argparse.Namespace:
    return build_parser(description=description, defaults=defaults).parse_args(argv)


def parse_and_prepare(
    argv: Sequence[str] | None = None,
    *,
    defaults: dict[str, object] | None = None,
    description: str = "Infer ChangeDINO on SAR scene tiles",
) -> argparse.Namespace:
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

    with torch.no_grad():
        pbar = tqdm(dataloader, total=len(dataloader), ncols=100, desc="Infer tiles")
        for batch_idx, batch in enumerate(pbar, start=1):
            img1 = batch["img1"].to(model.device, non_blocking=torch.cuda.is_available())
            img2 = batch["img2"].to(model.device, non_blocking=torch.cuda.is_available())
            logits = model.inference(img1, img2)
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy().astype(np.float32, copy=False)
            valid_masks = batch["valid_mask"].numpy().astype(np.float32, copy=False)
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
            pbar.set_postfix({"batch": batch_idx, "tiles": min(batch_idx * opt.batch_size, len(dataset))})

    valid_output = accum_weight > 0
    prob_map = np.full((full_height, full_width), PROB_NODATA, dtype=np.float32)
    prob_map[valid_output] = accum_prob[valid_output] / np.maximum(accum_weight[valid_output], 1e-6)

    binary_map = np.full((full_height, full_width), BINARY_NODATA, dtype=np.uint8)
    binary_map[valid_output] = (prob_map[valid_output] >= float(opt.threshold)).astype(np.uint8)

    prob_path = opt.output_dir / "change_prob.tif"
    binary_tif_path = opt.output_dir / "change_binary.tif"
    binary_png_path = opt.output_dir / "change_binary.png"
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
        "output_files": {
            "change_prob_tif": str(prob_path),
            "change_binary_tif": str(binary_tif_path),
            "change_binary_png": str(binary_png_path),
        },
        "source_image": str(source_pre),
        "source_shape": [full_height, full_width],
    }
    (opt.output_dir / "infer_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[DONE] tiles={len(dataset)} outputs={opt.output_dir}")


if __name__ == "__main__":
    main()
