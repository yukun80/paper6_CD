"""直方图与 Otsu 初始化工具。"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def extract_valid_values(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = array[mask]
    values = values[np.isfinite(values)]
    return values.astype(np.float64)


def compute_histogram(
    values: np.ndarray,
    bins: int,
    quantile_clip: Optional[Tuple[float, float]] = None,
) -> Dict[str, np.ndarray]:
    if values.size == 0:
        raise ValueError("空数组无法计算直方图")
    if quantile_clip is not None:
        q_low, q_high = quantile_clip
        lo = float(np.quantile(values, q_low))
        hi = float(np.quantile(values, q_high))
        if hi <= lo:
            lo, hi = float(values.min()), float(values.max())
    else:
        lo, hi = float(values.min()), float(values.max())
    if np.isclose(hi, lo):
        hi = lo + 1e-6

    counts, edges = np.histogram(values, bins=int(bins), range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts = counts.astype(np.float64)
    norm = counts / counts.sum() if counts.sum() > 0 else counts
    return {
        "counts": counts,
        "norm": norm,
        "edges": edges,
        "centers": centers,
        "range": np.array([lo, hi], dtype=np.float64),
    }


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    hist = compute_histogram(values, bins=bins)
    counts = hist["counts"]
    centers = hist["centers"]
    prob = counts / counts.sum()
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * centers)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    sigma_b = np.zeros_like(denom)
    valid = denom > 0
    sigma_b[valid] = (mu_t * omega[valid] - mu[valid]) ** 2 / denom[valid]
    idx = int(np.argmax(sigma_b))
    return float(centers[idx])


def empirical_pdf_from_mask(
    array: np.ndarray,
    select_mask: np.ndarray,
    bins: int,
    hist_range: Optional[Tuple[float, float]] = None,
) -> Dict[str, np.ndarray]:
    values = extract_valid_values(array, select_mask)
    if values.size == 0:
        raise ValueError("选区内无有效像元")
    if hist_range is None:
        lo, hi = float(values.min()), float(values.max())
    else:
        lo, hi = hist_range
    counts, edges = np.histogram(values, bins=bins, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    norm = counts.astype(np.float64)
    if norm.sum() > 0:
        norm /= norm.sum()
    return {"counts": counts.astype(np.float64), "norm": norm, "edges": edges, "centers": centers}
