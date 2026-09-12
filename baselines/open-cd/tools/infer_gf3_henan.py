#!/usr/bin/env python3
"""GF3 Henan 切片推理与整景拼接脚本。"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Iterable

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_opencd_gf3_henan_infer")

import numpy as np
from PIL import Image
import rasterio
from rich.progress import track
import torch

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
REPO_ROOT = PROJECT_ROOT.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from opencd.apis import OpenCDInferencer


PROB_NODATA = -1.0
BINARY_NODATA = 255

# 兼容 PyTorch 2.6+ 对 torch.load 默认开启 weights_only 的行为变化。
_ORIGINAL_TORCH_LOAD = torch.load

"""
cd /home/yukun/codes/paper6_waterlogging/baselines/open-cd
"""


def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _ORIGINAL_TORCH_LOAD(*args, **kwargs)


torch.load = _torch_load_compat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer GF3 Henan tiles with an Open-CD checkpoint")
    parser.add_argument("config", help="config file used to build the model")
    parser.add_argument("checkpoint", help="checkpoint path")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument(
        "--data-root",
        default="datasets/GF3_Henan_CD_infer",
        help="GF3 Henan infer dataset root, default: datasets/GF3_Henan_CD_infer",
    )
    parser.add_argument(
        "--work-dir",
        default="",
        help="model work dir, used as the default parent of output dirs",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="output directory for tile-level binary PNG predictions",
    )
    parser.add_argument(
        "--mosaic-dir",
        default="",
        help="output directory for stitched full-scene results",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="inference device, default: cuda:0",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="tile group size; effective forward batch size is always 1, default: 1",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="optional cap on number of tiles, 0 means all tiles",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="threshold applied on the stitched probability map, default: 0.5",
    )
    parser.add_argument(
        "--skip-mosaic",
        action="store_true",
        help="only save tile-level PNG predictions, skip full-scene stitching",
    )
    return parser.parse_args()


def default_output_dir(args: argparse.Namespace) -> Path:
    """模型和地区分目录，避免不同整景覆盖彼此。"""
    name = Path(args.data_root).name
    region = {"GF3_Henan_CD_infer": "Zhengzhou",
              "GF3_Zhuozhou_CD_infer": "Zhuozhou",
              "LT1_Guangxi_CD_infer": "Guangxi"}.get(name, name)
    return Path(args.output_root).resolve() / Path(args.config).stem / region


def ensure_tile_output_dir(args: argparse.Namespace) -> Path:
    """解析并创建切片输出目录。"""
    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        out_dir = default_output_dir(args) / "tile_png"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def ensure_mosaic_output_dir(args: argparse.Namespace) -> Path | None:
    """解析并创建整景输出目录。"""
    if args.skip_mosaic:
        return None

    if args.mosaic_dir:
        mosaic_dir = Path(args.mosaic_dir).resolve()
    else:
        mosaic_dir = default_output_dir(args) / "mosaic"
    mosaic_dir.mkdir(parents=True, exist_ok=True)
    return mosaic_dir


def load_tile_rows(data_root: Path) -> list[dict[str, str]]:
    """读取切片 manifest，并按 tile_index 排序。"""
    manifest_path = data_root / "tile_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")

    rows = sorted(rows, key=lambda row: int(row["tile_index"]))
    for row in rows:
        for key in ("a_png", "b_png", "valid_mask"):
            path = data_root / row[key]
            if not path.is_file():
                raise FileNotFoundError(f"Missing file referenced by manifest: {path}")
    return rows


def build_tile_inputs(
    data_root: Path,
    rows: list[dict[str, str]],
) -> list[tuple[str, str, Path, dict[str, str]]]:
    """将 manifest 行转换成推理输入对。"""
    records: list[tuple[str, str, Path, dict[str, str]]] = []
    a_prefix = Path("test") / "A"
    for row in rows:
        rel_path = Path(row["a_png"]).relative_to(a_prefix)
        records.append(
            (
                str((data_root / row["a_png"]).resolve()),
                str((data_root / row["b_png"]).resolve()),
                rel_path,
                row,
            )
        )
    return records


def batched(
    items: list[tuple[str, str, Path, dict[str, str]]], batch_size: int
) -> Iterable[list[tuple[str, str, Path, dict[str, str]]]]:
    """按固定批大小切分输入切片。"""
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def load_prepare_report(data_root: Path) -> dict[str, object]:
    """加载预处理报告，获取整景尺寸与源影像信息。"""
    report_path = data_root / "prepare_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Missing prepare report: {report_path}")
    return json.loads(report_path.read_text(encoding="utf-8"))


def resolve_source_path(path_str: str) -> Path:
    """兼容相对路径报告，恢复原始参考影像位置。"""
    path = Path(path_str)
    if path.is_absolute():
        return path

    candidates = [
        (REPO_ROOT / path).resolve(),
        (Path.cwd() / path).resolve(),
        (PROJECT_ROOT / path).resolve(),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return (REPO_ROOT / path).resolve()


def build_blend_weight(size: int) -> np.ndarray:
    """构建中心更高、边缘更低的 overlap 融合权重窗。"""
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
    """沿用原始影像的空间参考写出整景 GeoTIFF。"""
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
    """生成整景 RGB 预览图。"""
    rgb = np.zeros((binary_map.shape[0], binary_map.shape[1], 3), dtype=np.uint8)
    rgb[binary_map == 0] = np.array([40, 40, 40], dtype=np.uint8)
    rgb[binary_map == 1] = np.array([255, 255, 255], dtype=np.uint8)
    Image.fromarray(rgb, mode="RGB").save(out_path)


def save_binary_prediction(prediction, save_path: Path) -> None:
    """将 tile 预测结果保存为黑白 PNG。"""
    pred_mask = prediction.pred_sem_seg.data.squeeze().detach().cpu().numpy()
    pred_mask = (pred_mask > 0).astype(np.uint8) * 255
    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pred_mask, mode="L").save(save_path)


def extract_change_probability(prediction) -> np.ndarray:
    """从 Open-CD 输出中提取 change 类概率图。"""
    seg_logits = prediction.seg_logits.data.detach().cpu()
    if seg_logits.ndim != 3:
        raise ValueError(f"Unexpected seg_logits shape: {tuple(seg_logits.shape)}")

    if seg_logits.shape[0] == 1:
        # Open-CD 的单通道推理结果在 postprocess_result 中已做过 sigmoid，
        # 这里直接取概率图，避免重复激活后整图阈值化全白。
        prob = seg_logits[0]
    else:
        prob = torch.softmax(seg_logits, dim=0)[1]
    return prob.numpy().astype(np.float32, copy=False)


def accumulate_tile_probability(
    data_root: Path,
    row: dict[str, str],
    prob_tile: np.ndarray,
    base_weight: np.ndarray,
    accum_prob: np.ndarray,
    accum_weight: np.ndarray,
) -> None:
    """将单个切片概率图按 valid_mask 与权重窗累积回整景。"""
    top = int(row["top"])
    left = int(row["left"])
    height = int(row["height"])
    width = int(row["width"])

    with rasterio.open(data_root / row["valid_mask"]) as ds_mask:
        valid_mask = ds_mask.read(1).astype(np.float32, copy=False)

    weight = base_weight[:height, :width] * valid_mask[:height, :width]
    if not np.any(weight > 0):
        return

    accum_prob[top : top + height, left : left + width] += prob_tile[:height, :width] * weight
    accum_weight[top : top + height, left : left + width] += weight


def finalize_mosaic_outputs(
    prepare_report: dict[str, object],
    mosaic_dir: Path,
    accum_prob: np.ndarray,
    accum_weight: np.ndarray,
    threshold: float,
    *,
    data_root: Path,
    config_path: Path,
    checkpoint_path: Path,
    tile_output_dir: Path,
    total_tiles: int,
    used_tiles: int,
) -> None:
    """将累积结果写成整景概率图、二值图和报告。"""
    source = prepare_report["source"]
    source_pre = resolve_source_path(str(source["pre_image"]))
    valid_output = accum_weight > 0

    prob_map = np.full(accum_prob.shape, PROB_NODATA, dtype=np.float32)
    prob_map[valid_output] = accum_prob[valid_output] / np.maximum(accum_weight[valid_output], 1e-6)

    binary_map = np.full(accum_prob.shape, BINARY_NODATA, dtype=np.uint8)
    binary_map[valid_output] = (prob_map[valid_output] >= float(threshold)).astype(np.uint8)

    prob_path = mosaic_dir / "change_prob.tif"
    binary_tif_path = mosaic_dir / "change_binary.tif"
    binary_png_path = mosaic_dir / "change_binary.png"
    report_path = mosaic_dir / "infer_report.json"

    print("[INFO] Writing stitched outputs ...")
    write_geotiff(source_pre, prob_path, prob_map, "float32", PROB_NODATA)
    write_geotiff(source_pre, binary_tif_path, binary_map, "uint8", BINARY_NODATA)
    write_preview_png(binary_map, binary_png_path)

    report = {
        "tiles_root": str(data_root),
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "tile_output_dir": str(tile_output_dir),
        "threshold": float(threshold),
        "partial_run": used_tiles < total_tiles,
        "used_tiles": used_tiles,
        "total_tiles": total_tiles,
        "source_image": str(source_pre),
        "source_shape": [int(source["height"]), int(source["width"])],
        "output_files": {
            "change_prob_tif": str(prob_path),
            "change_binary_tif": str(binary_tif_path),
            "change_binary_png": str(binary_png_path),
        },
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def predict_single_tiles(inferencer, inputs: list) -> list:
    """底层预处理器仅支持单切片；严格校验数量防止 zip 静默丢失。"""
    predictions = []
    for processed in inferencer.preprocess(inputs, batch_size=1):
        result = list(inferencer.forward(processed))
        if len(result) != 1:
            raise RuntimeError(f"Expected one prediction per forward, got {len(result)}")
        predictions.extend(result)
    if len(predictions) != len(inputs):
        raise RuntimeError(f"Prediction count mismatch: {len(predictions)} != {len(inputs)}")
    return predictions


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0.")
    if args.batch_size > 1:
        print(f"[INFO] Requested group size {args.batch_size}; effective forward batch size is 1.")
    os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / "pretrained/torch"))
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be within [0, 1].")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested device {args.device} is unavailable, fallback to cpu.")
        args.device = "cpu"

    data_root = Path(args.data_root).resolve()
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    tile_output_dir = ensure_tile_output_dir(args)
    mosaic_dir = ensure_mosaic_output_dir(args)

    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    all_rows = load_tile_rows(data_root)
    prepare_report = load_prepare_report(data_root)
    total_tiles = len(all_rows)
    if args.limit > 0:
        all_rows = all_rows[: args.limit]

    tile_records = build_tile_inputs(data_root, all_rows)
    print(f"Config: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Data root: {data_root}")
    print(f"Tile output dir: {tile_output_dir}")
    print(f"Mosaic output dir: {mosaic_dir if mosaic_dir is not None else '<skipped>'}")
    print(f"Tiles used: {len(tile_records)} / {total_tiles}")
    print(f"Batch size: {args.batch_size}")
    print(f"Threshold: {args.threshold}")
    print(f"Device: {args.device}")

    inferencer = OpenCDInferencer(
        model=str(config_path),
        weights=str(checkpoint_path),
        device=args.device,
        scope="opencd",
    )

    accum_prob = None
    accum_weight = None
    base_weight = None
    if mosaic_dir is not None:
        source = prepare_report["source"]
        full_height = int(source["height"])
        full_width = int(source["width"])
        tile_size = int(prepare_report["params"]["tile_size"])
        accum_prob = np.zeros((full_height, full_width), dtype=np.float32)
        accum_weight = np.zeros((full_height, full_width), dtype=np.float32)
        base_weight = build_blend_weight(tile_size)

    total_saved = 0
    for batch in track(list(batched(tile_records, args.batch_size)), description="GF3 inference"):
        inputs = [[pre_path, post_path] for pre_path, post_path, _, _ in batch]
        predictions = predict_single_tiles(inferencer, inputs)

        for (_, _, rel_path, row), prediction in zip(batch, predictions):
            save_binary_prediction(prediction, tile_output_dir / rel_path)
            total_saved += 1

            if mosaic_dir is not None:
                prob_tile = extract_change_probability(prediction)
                accumulate_tile_probability(
                    data_root,
                    row,
                    prob_tile,
                    base_weight,
                    accum_prob,
                    accum_weight,
                )

    if mosaic_dir is not None:
        finalize_mosaic_outputs(
            prepare_report,
            mosaic_dir,
            accum_prob,
            accum_weight,
            args.threshold,
            data_root=data_root,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            tile_output_dir=tile_output_dir,
            total_tiles=total_tiles,
            used_tiles=len(tile_records),
        )

    print(f"Saved tile PNGs: {total_saved}")


if __name__ == "__main__":
    main()
