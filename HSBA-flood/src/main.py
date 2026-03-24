"""HSBA-flood 运行入口。"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .evaluate import evaluate_binary_mask
from .flood_mapping import build_flood_products, fit_global_bm_models
from .hsba import ArrayTileSource, DifferenceTileSource, run_hsba_with_fallback
from .io_utils import (
    ensure_dir,
    load_yaml,
    save_json,
    save_yaml,
    setup_logger,
    write_mask_raster,
    write_validated_difference_raster,
    write_validated_float_raster,
)
from .preprocess import prepare_input_pair
from .visualization import (
    save_difference_preview,
    save_float_preview,
    save_flood_overlay,
    save_hist_fit_plot,
    save_mask_preview,
    save_rmse_heatmap,
)

"""
python -m src.main --config configs/s1_henan.yaml
python -m src.main --config configs/gf3_henan.yaml
python -m src.main --config configs/gf3_henan.yaml --config configs/s1_henan.yaml
"""

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HSBA-flood unsupervised SAR flood mapping")
    parser.add_argument("--config", action="append", required=True, help="YAML 配置路径，可重复传入多个")
    return parser.parse_args()


def _save_intermediate_rasters(
    output_dir: Path,
    prepared: Dict[str, Any],
    hsba_results: Dict[str, Any],
    flood: Dict[str, Any],
    config: Dict[str, Any],
) -> None:
    profile = prepared["profile"]
    valid_mask = prepared["valid_mask"]
    write_mask_raster(output_dir / "valid_mask.tif", valid_mask.astype(np.uint8), profile, valid_mask=valid_mask)
    if bool(config.get("export_full_valid_rasters", True)):
        write_validated_float_raster(output_dir / "XF_valid.tif", prepared["XF"], profile, valid_mask, nodata=np.nan)
        write_validated_float_raster(output_dir / "XR_valid.tif", prepared["XR"], profile, valid_mask, nodata=np.nan)
        write_validated_difference_raster(
            output_dir / "XC_valid.tif",
            prepared["XR"],
            prepared["XF"],
            profile,
            valid_mask,
            nodata=np.nan,
        )
    write_mask_raster(output_dir / "BM_F.tif", hsba_results["BM_F"], profile, valid_mask=valid_mask)
    write_mask_raster(output_dir / "BM_C.tif", hsba_results["BM_C"], profile, valid_mask=valid_mask)
    write_mask_raster(output_dir / "BM_intersection.tif", hsba_results["BM"], profile, valid_mask=valid_mask)
    write_mask_raster(output_dir / "post_event_water_mask.tif", flood["post_event_water"], profile, valid_mask=valid_mask)
    write_mask_raster(output_dir / "pre_event_water_mask.tif", flood["pre_event_water"], profile, valid_mask=valid_mask)
    write_mask_raster(output_dir / "final_flood_map.tif", flood["final_flood"], profile, valid_mask=valid_mask)


def _save_pngs(
    output_dir: Path,
    prepared: Dict[str, Any],
    hsba_results: Dict[str, Any],
    flood: Dict[str, Any],
    fits: Dict[str, Any],
    config: Dict[str, Any],
) -> None:
    valid_mask = prepared["valid_mask"]
    preview_max_pixels = int(config.get("preview_max_pixels", 2_000_000))
    preview_stride = config.get("preview_downsample_stride")
    preview_stride = None if preview_stride in (None, "null") else int(preview_stride)

    save_float_preview(
        output_dir / "xf_preview.png",
        prepared["XF"],
        valid_mask,
        "XF Preview",
        max_pixels=preview_max_pixels,
        stride=preview_stride,
    )
    save_float_preview(
        output_dir / "xr_preview.png",
        prepared["XR"],
        valid_mask,
        "XR Preview",
        max_pixels=preview_max_pixels,
        stride=preview_stride,
    )
    save_difference_preview(
        output_dir / "xc_preview.png",
        prepared["XR"],
        prepared["XF"],
        valid_mask,
        "XC Preview",
        max_pixels=preview_max_pixels,
        stride=preview_stride,
    )
    save_mask_preview(
        output_dir / "bm_f_preview.png",
        hsba_results["BM_F"],
        valid_mask,
        "BM_F",
        max_pixels=preview_max_pixels,
        stride=preview_stride,
    )
    save_mask_preview(
        output_dir / "bm_c_preview.png",
        hsba_results["BM_C"],
        valid_mask,
        "BM_C",
        max_pixels=preview_max_pixels,
        stride=preview_stride,
    )
    save_mask_preview(
        output_dir / "bm_intersection_preview.png",
        hsba_results["BM"],
        valid_mask,
        "BM Intersection",
        max_pixels=preview_max_pixels,
        stride=preview_stride,
    )

    save_hist_fit_plot(
        output_dir / "hist_fit_xf_bm.png",
        fits["xf_fit"],
        "XF[BM] Histogram + Double Gaussian",
        left_label="Gw",
        right_label="Gnw",
        extra_lines={
            "sigma_seed": flood["summary"]["sigma_seed"],
            "intersection": flood["summary"]["threshold_intersection"],
        },
    )
    save_hist_fit_plot(
        output_dir / "hist_fit_xc_bm.png",
        fits["xc_fit"],
        "XC[BM] Histogram + Double Gaussian",
        left_label="Gnc",
        right_label="Gc",
        extra_lines={"delta_sigma_CD_best": flood["summary"]["delta_sigma_cd_best"]},
    )
    save_rmse_heatmap(
        output_dir / "threshold_search_rmse_heatmap.png",
        flood["search"]["rmse_grid"],
        flood["search"]["sigma_rg_grid"],
        flood["search"]["delta_cd_grid"],
        flood["summary"]["sigma_rg_best"],
        flood["summary"]["delta_sigma_cd_best"],
    )
    save_flood_overlay(
        output_dir / "final_flood_overlay.png",
        prepared["XF"],
        valid_mask,
        flood["final_flood"],
        max_pixels=preview_max_pixels,
        stride=preview_stride,
    )


def run_single_config(config_path: str | Path) -> None:
    config_path = Path(config_path).resolve()
    config = load_yaml(config_path)
    config_dir = config_path.parent
    output_dir = ensure_dir((config_dir / config["output_dir"]).resolve())
    logger = setup_logger(output_dir, verbose=bool(config.get("verbose_logging", True)))
    logger.info("开始运行数据集: %s", config.get("dataset_name", output_dir.name))
    save_yaml(config, output_dir / "config_used.yaml")

    try:
        prepared = prepare_input_pair(config, config_dir, logger)
        hsba_f = run_hsba_with_fallback(ArrayTileSource(prepared["XF"]), prepared["valid_mask"], config, logger, "XF")
        hsba_c = run_hsba_with_fallback(
            DifferenceTileSource(prepared["XR"], prepared["XF"]),
            prepared["valid_mask"],
            config,
            logger,
            "XC",
        )
        bm = hsba_f["mask"] & hsba_c["mask"] & prepared["valid_mask"]
        bm_pixels = int(bm.sum())
        if bm_pixels < int(config.get("bm_min_pixels", 4096)):
            raise RuntimeError(f"BM 交集像元过少: {bm_pixels}")

        fits = fit_global_bm_models(prepared["XF"], prepared["XR"], bm, config)
        flood = build_flood_products(
            xf=prepared["XF"],
            xr=prepared["XR"],
            valid_mask=prepared["valid_mask"],
            bm_mask=bm,
            fits=fits,
            config=config,
        )
        hsba_results = {"BM_F": hsba_f["mask"], "BM_C": hsba_c["mask"], "BM": bm}

        if bool(config.get("save_intermediate_tifs", True)):
            _save_intermediate_rasters(output_dir, prepared, hsba_results, flood, config)
        if bool(config.get("save_intermediate_pngs", True)):
            _save_pngs(output_dir, prepared, hsba_results, flood, fits, config)

        eval_result = evaluate_binary_mask(flood["final_flood"], config.get("gt_path"), prepared["valid_mask"])
        hsba_summary = {
            "xf_level_stats": hsba_f["level_stats"],
            "xc_level_stats": hsba_c["level_stats"],
            "xf_selected_tile_count": hsba_f["selected_tile_count"],
            "xc_selected_tile_count": hsba_c["selected_tile_count"],
        }
        if bool(config.get("save_selected_tile_details", False)):
            hsba_summary["xf_selected_tiles"] = hsba_f["selected_tiles"]
            hsba_summary["xc_selected_tiles"] = hsba_c["selected_tiles"]

        summary = {
            "dataset_name": config.get("dataset_name"),
            "input": {"xr": prepared["xr_meta"], "xf": prepared["xf_meta"]},
            "valid_stats": prepared["valid_stats"],
            "bm_stats": {
                "bm_f_pixels": int(hsba_f["mask"].sum()),
                "bm_c_pixels": int(hsba_c["mask"].sum()),
                "bm_intersection_pixels": int(bm.sum()),
                "bm_intersection_ratio": float(bm.sum() / prepared["valid_stats"]["valid_pixels"]),
            },
            "hsba": hsba_summary,
            "fits": flood["fits"],
            "thresholds": flood["summary"],
            "evaluation": eval_result,
        }
        params = {
            "thresholds": flood["summary"],
            "xf_fit": flood["fits"]["xf_fit"],
            "xc_fit": flood["fits"]["xc_fit"],
            "search_space": {
                "sigma_rg_grid": flood["search"]["sigma_rg_grid"].tolist(),
                "delta_cd_grid": flood["search"]["delta_cd_grid"].tolist(),
            },
        }
        save_json(summary, output_dir / "summary_metrics.json")
        save_json(params, output_dir / "fitted_params.json")
        logger.info("完成: dataset=%s final_flood_pixels=%d", config.get("dataset_name"), int(flood["final_flood"].sum()))
    except Exception as exc:  # noqa: BLE001
        logger.error("运行失败: %s", exc)
        logger.debug(traceback.format_exc())
        raise


def main() -> None:
    args = parse_args()
    for config_path in args.config:
        run_single_config(config_path)


if __name__ == "__main__":
    main()
