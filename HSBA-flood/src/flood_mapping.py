"""从 BM 拟合到最终洪水图的主流程。"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .gaussian_fit import fit_double_gaussian
from .hist_utils import extract_valid_values
from .region_growing import region_grow_from_seeds, remove_small_components
from .threshold_search import estimate_seed_threshold, search_best_thresholds


def fit_global_bm_models(
    xf: np.ndarray,
    xr: np.ndarray,
    bm_mask: np.ndarray,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    quantile_clip = tuple(config["histogram_quantile_clip"]) if config.get("histogram_quantile_clip") else None
    xf_values = extract_valid_values(xf, bm_mask)
    xc_values = (xr[bm_mask] - xf[bm_mask]).astype(np.float64, copy=False)
    xf_fit = fit_double_gaussian(xf_values, bins=int(config["hist_bins"]), quantile_clip=quantile_clip)
    xc_fit = fit_double_gaussian(xc_values, bins=int(config["hist_bins"]), quantile_clip=quantile_clip)
    if not xf_fit.success:
        raise RuntimeError(f"XF[BM] 双高斯拟合失败: {xf_fit.message}")
    if not xc_fit.success:
        raise RuntimeError(f"XC[BM] 双高斯拟合失败: {xc_fit.message}")
    return {"xf_fit": xf_fit, "xc_fit": xc_fit}


def build_flood_products(
    xf: np.ndarray,
    xr: np.ndarray,
    valid_mask: np.ndarray,
    bm_mask: np.ndarray,
    fits: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    xf_fit = fits["xf_fit"]
    xc_fit = fits["xc_fit"]
    connectivity = int(config.get("rg_connectivity", 8))
    min_component_pixels = int(config.get("min_component_pixels", 9))

    seed_info = estimate_seed_threshold(xf, bm_mask, xf_fit, float(config["seed_tau"]))
    search = search_best_thresholds(xf, xr, valid_mask, bm_mask, float(seed_info["sigma_seed"]), xf_fit, config)

    permanent_seed = (xr <= float(seed_info["sigma_seed"])) & valid_mask
    permanent_candidate = (xr <= float(search["sigma_rg_best"])) & valid_mask
    pre_event_water_raw = region_grow_from_seeds(
        permanent_seed,
        permanent_candidate,
        connectivity=connectivity,
    )
    pre_event_water = remove_small_components(
        pre_event_water_raw,
        min_pixels=min_component_pixels,
        connectivity=connectivity,
    )

    post_event_water_raw = search["rg_mask"]
    post_event_water = remove_small_components(
        post_event_water_raw,
        min_pixels=min_component_pixels,
        connectivity=connectivity,
    )

    # 最终洪水先扣除后处理后的灾前单时相水体，再做最终连通域清理。
    final_flood_raw = search["cd_mask"] & (~pre_event_water) & valid_mask
    final_flood = remove_small_components(
        final_flood_raw,
        min_pixels=min_component_pixels,
        connectivity=connectivity,
    )

    return {
        "seed_mask": search["seed_mask"],
        "post_event_water_raw": post_event_water_raw,
        "post_event_water": post_event_water,
        "cd_mask_raw": search["cd_mask"],
        "pre_event_water_raw": pre_event_water_raw,
        "pre_event_water": pre_event_water,
        "final_flood_raw": final_flood_raw,
        "final_flood": final_flood,
        "seed_info": seed_info,
        "search": search,
        "fits": {
            "xf_fit": xf_fit.to_dict(),
            "xc_fit": xc_fit.to_dict(),
        },
        "summary": {
            "sigma_seed": float(seed_info["sigma_seed"]),
            "threshold_intersection": seed_info["threshold_intersection"],
            "sigma_rg_best": float(search["sigma_rg_best"]),
            "delta_sigma_cd_best": float(search["delta_sigma_cd_best"]),
            "rmse_best": float(search["rmse_best"]),
            "seed_pixels": int(search["seed_mask"].sum()),
            "post_event_water_raw_pixels": int(post_event_water_raw.sum()),
            "post_event_water_pixels": int(post_event_water.sum()),
            "cd_mask_raw_pixels": int(search["cd_mask"].sum()),
            "pre_event_water_raw_pixels": int(pre_event_water_raw.sum()),
            "pre_event_water_pixels": int(pre_event_water.sum()),
            "final_flood_raw_pixels": int(final_flood_raw.sum()),
            "final_flood_pixels": int(final_flood.sum()),
        },
    }
