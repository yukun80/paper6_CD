"""种子阈值估计与 RG/CD 联合网格搜索。"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .gaussian_fit import DoubleGaussianFitResult, gaussian, gaussian_intersection
from .hist_utils import empirical_pdf_from_mask
from .region_growing import region_grow_from_seeds


def estimate_seed_threshold(
    xf: np.ndarray,
    bm_mask: np.ndarray,
    fit_result: DoubleGaussianFitResult,
    seed_tau: float,
) -> Dict[str, Optional[float]]:
    hist = empirical_pdf_from_mask(xf, bm_mask, bins=fit_result.bins, hist_range=fit_result.hist_range)
    gw = gaussian(hist["centers"], fit_result.left.amplitude, fit_result.left.mu, fit_result.left.sd)
    gw_norm = gw / gw.sum() if gw.sum() > 0 else np.zeros_like(gw)
    delta = np.abs(hist["norm"] - gw_norm)
    indices = np.where(delta > float(seed_tau))[0]
    sigma_seed = float(hist["centers"][indices[0]]) if indices.size > 0 else float(fit_result.left.mu)
    return {
        "sigma_seed": sigma_seed,
        "threshold_intersection": gaussian_intersection(fit_result.left, fit_result.right),
        "hist_centers": hist["centers"],
        "hist_norm": hist["norm"],
        "gw_norm": gw_norm,
        "abs_diff": delta,
    }


def _build_sigma_rg_grid(
    sigma_seed: float,
    fit_result: DoubleGaussianFitResult,
    xf_bm_values: np.ndarray,
    step: float,
) -> np.ndarray:
    upper = min(
        float(fit_result.right.mu),
        float(fit_result.left.mu + 4.0 * fit_result.left.sd),
        float(np.quantile(xf_bm_values, 0.99)),
    )
    upper = max(upper, sigma_seed + step)
    return np.arange(sigma_seed, upper + 0.5 * step, step, dtype=np.float64)


def _build_delta_cd_grid(config: Dict[str, Any]) -> np.ndarray:
    step = float(config["delta_cd_step"])
    start = float(config["delta_cd_min"])
    end = float(config["delta_cd_max"])
    return np.arange(start, end + 0.5 * step, step, dtype=np.float64)


def _rmse_to_theoretical_water_values(
    xf_values: np.ndarray,
    fit_result: DoubleGaussianFitResult,
) -> float:
    if xf_values.size == 0:
        return float("inf")
    counts, edges = np.histogram(xf_values, bins=fit_result.bins, range=fit_result.hist_range)
    hist_norm = counts.astype(np.float64)
    if hist_norm.sum() > 0:
        hist_norm /= hist_norm.sum()
    centers = 0.5 * (edges[:-1] + edges[1:])
    gw = gaussian(centers, fit_result.left.amplitude, fit_result.left.mu, fit_result.left.sd)
    gw_norm = gw / gw.sum() if gw.sum() > 0 else np.zeros_like(gw)
    return float(np.sqrt(np.mean((hist_norm - gw_norm) ** 2)))


def search_best_thresholds(
    xf: np.ndarray,
    xr: np.ndarray,
    valid_mask: np.ndarray,
    bm_mask: np.ndarray,
    sigma_seed: float,
    xf_fit: DoubleGaussianFitResult,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    bm_index = bm_mask & valid_mask & np.isfinite(xf) & np.isfinite(xr)
    xf_bm_values = xf[bm_index]
    if xf_bm_values.size == 0:
        raise RuntimeError("BM 区域无有效 XF 像元，无法执行阈值搜索")

    sigma_rg_grid = _build_sigma_rg_grid(sigma_seed, xf_fit, xf_bm_values, float(config["sigma_rg_step"]))
    delta_cd_grid = _build_delta_cd_grid(config)
    if sigma_rg_grid.size == 0 or delta_cd_grid.size == 0:
        raise RuntimeError("阈值搜索空间为空")

    seed_mask = (xf <= sigma_seed) & valid_mask
    rmse_grid = np.full((sigma_rg_grid.size, delta_cd_grid.size), np.inf, dtype=np.float64)
    best: Dict[str, Any] = {
        "rmse": float("inf"),
        "sigma_rg": None,
        "delta_sigma_cd": None,
        "seed_mask": seed_mask,
    }

    connectivity = int(config.get("rg_connectivity", 8))
    bm_xf_values = xf[bm_index]
    bm_xc_values = xr[bm_index] - xf[bm_index]
    for i, sigma_rg in enumerate(sigma_rg_grid):
        candidate = (xf <= sigma_rg) & valid_mask
        rg_mask = region_grow_from_seeds(seed_mask, candidate, connectivity=connectivity)
        rg_mask_bm = rg_mask[bm_index]
        rg_xf_values = bm_xf_values[rg_mask_bm]
        rg_xc_values = bm_xc_values[rg_mask_bm]
        for j, delta_cd in enumerate(delta_cd_grid):
            rmse = _rmse_to_theoretical_water_values(rg_xf_values[rg_xc_values >= delta_cd], xf_fit)
            rmse_grid[i, j] = rmse
            if rmse < best["rmse"]:
                best.update(
                    {
                        "rmse": float(rmse),
                        "sigma_rg": float(sigma_rg),
                        "delta_sigma_cd": float(delta_cd),
                    }
                )

    if best["sigma_rg"] is None or best["delta_sigma_cd"] is None:
        raise RuntimeError("搜索空间存在，但没有找到有效阈值组合")

    best_candidate = (xf <= float(best["sigma_rg"])) & valid_mask
    best_rg_mask = region_grow_from_seeds(seed_mask, best_candidate, connectivity=connectivity)
    best_cd_mask = np.zeros_like(valid_mask, dtype=bool)
    best_cd_mask[valid_mask] = (xr[valid_mask] - xf[valid_mask]) >= float(best["delta_sigma_cd"])
    best_cd_mask &= best_rg_mask

    return {
        "seed_mask": best["seed_mask"],
        "rg_mask": best_rg_mask,
        "cd_mask": best_cd_mask,
        "sigma_rg_grid": sigma_rg_grid,
        "delta_cd_grid": delta_cd_grid,
        "rmse_grid": rmse_grid,
        "sigma_rg_best": best["sigma_rg"],
        "delta_sigma_cd_best": best["delta_sigma_cd"],
        "rmse_best": best["rmse"],
    }
