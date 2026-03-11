"""GF3_Henan 从预处理到全图回拼的一体化入口。"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from .config import load_config
from .data import GF3TileDataset
from .gf3_preprocess import prepare_gf3_pair
from .optimizer import PromptOptimizationEnv, rule_greedy_optimize
from .prompts import PromptCandidateGenerator
from .segmenter import build_segmenter


def parse_args():
    parser = argparse.ArgumentParser("Run GF3_Henan prompt optimization pipeline")
    parser.add_argument("--config-file", type=str, default="sar_prompt_flood/config/gf3_henan.json")
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--preprocess-only", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def main():
    args = parse_args()
    cfg = load_config(args.config_file)
    set_seed(int(cfg["runtime"].get("seed", 42)))

    pair_root = Path(cfg["data"]["pair_root"])
    pre_path = pair_root / cfg["data"]["pre_name"]
    post_path = pair_root / cfg["data"]["post_name"]
    processed_root = Path(cfg["data"]["processed_root"])
    manifest_path = processed_root / "manifest.json"
    if args.force_preprocess or not manifest_path.exists():
        prepare_gf3_pair(
            pre_path=str(pre_path),
            post_path=str(post_path),
            out_root=str(processed_root),
            tile_size=cfg["preprocess"].get("tile_size", 512),
            overlap=cfg["preprocess"].get("overlap", 128),
            min_valid_ratio=cfg["preprocess"].get("min_valid_ratio", 0.6),
            clip_percentiles=tuple(cfg["preprocess"].get("clip_percentiles", [1.0, 99.0])),
            max_tiles=cfg["preprocess"].get("max_tiles", -1),
        )
    if args.preprocess_only:
        print(f"Prepared GF3 dataset at {processed_root}")
        return

    dataset = GF3TileDataset(str(processed_root), max_tiles=cfg["inference"].get("max_tiles", -1))
    prompt_generator = PromptCandidateGenerator(**cfg["prompts"])
    sam_segmenter = build_segmenter(cfg)

    out_dir = Path(cfg["runtime"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    tile_vis_dir = out_dir / "vis"
    tile_vis_dir.mkdir(parents=True, exist_ok=True)

    height = int(dataset.reference["height"])
    width = int(dataset.reference["width"])
    acc_init = np.zeros((height, width), dtype=np.float32)
    acc_opt = np.zeros((height, width), dtype=np.float32)
    acc_final = np.zeros((height, width), dtype=np.float32)
    acc_change = np.zeros((height, width), dtype=np.float32)
    acc_weight = np.zeros((height, width), dtype=np.float32)
    records = []

    for sample in tqdm(dataset, desc="GF3 tiles"):
        prompt_set = prompt_generator.generate(sample)
        env = PromptOptimizationEnv(
            prompt_set=prompt_set,
            segmenter=sam_segmenter,
            max_steps=cfg["optimizer"].get("max_steps", 10),
            min_positive_points=cfg["optimizer"].get("min_positive_points", 2),
            max_positive_points=cfg["optimizer"].get("max_positive_points", 10),
            min_negative_points=cfg["optimizer"].get("min_negative_points", 2),
            max_negative_points=cfg["optimizer"].get("max_negative_points", 10),
        )
        init_summary = env.export_summary()
        opt_summary = rule_greedy_optimize(env)
        final_mask = opt_summary.mask

        top = int(sample["meta"]["row_off"])
        left = int(sample["meta"]["col_off"])
        h = int(sample["meta"]["height"])
        w = int(sample["meta"]["width"])
        window = np.s_[top : top + h, left : left + w]
        weight = sample["valid_mask"].numpy().astype(np.float32)
        acc_init[window] += init_summary.mask.astype(np.float32) * weight
        acc_opt[window] += opt_summary.mask.astype(np.float32) * weight
        acc_final[window] += final_mask.astype(np.float32) * weight
        acc_change[window] += sample["change_score"].numpy().astype(np.float32) * weight
        acc_weight[window] += weight

        if cfg["inference"].get("save_tile_visuals", True):
            save_tile_visual(tile_vis_dir / f"{sample['tile_id']}.png", prompt_set.pseudo_rgb, init_summary, opt_summary, final_mask)

        records.append(
            {
                "tile_id": sample["tile_id"],
                "selected_backend": "sam",
                "low_confidence": prompt_set.low_confidence or bool(sample["meta"]["low_confidence"]),
                "init_metrics": init_summary.metrics,
                "opt_metrics": opt_summary.metrics,
                "actions": opt_summary.action_history,
                "init_mask_area_ratio": float(init_summary.mask.mean()),
                "opt_mask_area_ratio": float(opt_summary.mask.mean()),
            }
        )

    weight_safe = np.clip(acc_weight, a_min=1e-6, a_max=None)
    init_prob = acc_init / weight_safe
    opt_prob = acc_opt / weight_safe
    final_prob = acc_final / weight_safe
    change_prob = acc_change / weight_safe
    final_mask = (final_prob >= float(cfg["inference"].get("mosaic_threshold", 0.5))).astype(np.uint8)
    uncertainty = 1.0 - np.abs(2.0 * np.clip(final_prob, 0.0, 1.0) - 1.0)

    save_mosaic_geotiff(processed_root / "manifest.json", out_dir / "change_score.tif", change_prob.astype(np.float32), nodata=0.0)
    save_mosaic_geotiff(processed_root / "manifest.json", out_dir / "init_mask.tif", init_prob.astype(np.float32), nodata=0.0)
    save_mosaic_geotiff(processed_root / "manifest.json", out_dir / "opt_mask.tif", opt_prob.astype(np.float32), nodata=0.0)
    save_mosaic_geotiff(processed_root / "manifest.json", out_dir / "final_mask.tif", final_mask.astype(np.uint8), nodata=0)
    save_mosaic_geotiff(processed_root / "manifest.json", out_dir / "uncertainty.tif", uncertainty.astype(np.float32), nodata=0.0)

    summary = {
        "num_tiles": len(records),
        "selected_backend": "sam",
        "low_confidence_tiles": int(sum(1 for x in records if x["low_confidence"])),
        "mean_init_area_ratio": float(init_prob.mean()),
        "mean_opt_area_ratio": float(opt_prob.mean()),
        "mean_final_area_ratio": float(final_mask.mean()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "tile_metrics.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def save_tile_visual(path: Path, pseudo_rgb: np.ndarray, init_summary, opt_summary, final_mask: np.ndarray) -> None:
    image = Image.fromarray(pseudo_rgb.astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    for x, y in init_summary.pos_points:
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(0, 255, 0))
    for x, y in init_summary.neg_points:
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(255, 0, 255))
    image.save(path)


def save_mosaic_geotiff(manifest_path: Path, out_path: Path, arr: np.ndarray, nodata) -> None:
    import rasterio
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    ref = manifest["reference"]
    transform = rasterio.Affine(*ref["transform"])
    profile = {
        "driver": "GTiff",
        "height": arr.shape[0],
        "width": arr.shape[1],
        "count": 1,
        "dtype": str(arr.dtype),
        "crs": ref["crs"],
        "transform": transform,
        "compress": "deflate",
        "predictor": 2 if arr.dtype != np.uint8 else 1,
        "nodata": nodata,
    }
    with rasterio.open(out_path, "w", **profile) as ds:
        ds.write(arr, 1)


if __name__ == "__main__":
    main()
