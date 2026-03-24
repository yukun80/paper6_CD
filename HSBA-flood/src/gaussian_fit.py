"""双高斯拟合与统计判据。"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit

from .hist_utils import compute_histogram, otsu_threshold


@dataclass
class GaussianParams:
    amplitude: float
    mu: float
    sd: float


@dataclass
class DoubleGaussianFitResult:
    success: bool
    left: GaussianParams
    right: GaussianParams
    ashman_d: float
    bc: float
    surface_ratio: float
    hist_range: Tuple[float, float]
    bins: int
    centers: np.ndarray
    hist_counts: np.ndarray
    hist_norm: np.ndarray
    fit_curve: np.ndarray
    fit_norm: np.ndarray
    otsu_init: float
    message: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "success": self.success,
            "left": asdict(self.left),
            "right": asdict(self.right),
            "ashman_d": self.ashman_d,
            "bc": self.bc,
            "surface_ratio": self.surface_ratio,
            "hist_range": [float(self.hist_range[0]), float(self.hist_range[1])],
            "bins": self.bins,
            "otsu_init": self.otsu_init,
            "message": self.message,
        }


def gaussian(x: np.ndarray, amplitude: float, mu: float, sd: float) -> np.ndarray:
    sd = max(float(sd), 1e-6)
    return amplitude * np.exp(-((x - mu) ** 2) / (2.0 * sd**2))


def double_gaussian(x: np.ndarray, a1: float, mu1: float, sd1: float, a2: float, mu2: float, sd2: float) -> np.ndarray:
    return gaussian(x, a1, mu1, sd1) + gaussian(x, a2, mu2, sd2)


def _prepare_initial_guess(values: np.ndarray, centers: np.ndarray, counts: np.ndarray, otsu_value: float) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    left_values = values[values <= otsu_value]
    right_values = values[values > otsu_value]
    if left_values.size < 2 or right_values.size < 2:
        raise ValueError("Otsu 初始化后某一侧有效像元过少")

    mu1, mu2 = float(left_values.mean()), float(right_values.mean())
    sd1, sd2 = float(left_values.std(ddof=0)), float(right_values.std(ddof=0))
    sd1 = max(sd1, 1e-3)
    sd2 = max(sd2, 1e-3)
    idx1 = int(np.argmin(np.abs(centers - mu1)))
    idx2 = int(np.argmin(np.abs(centers - mu2)))
    a1 = max(float(counts[idx1]), 1e-6)
    a2 = max(float(counts[idx2]), 1e-6)

    p0 = np.array([a1, mu1, sd1, a2, mu2, sd2], dtype=np.float64)
    lower = np.array([1e-6, values.min(), 1e-6, 1e-6, values.min(), 1e-6], dtype=np.float64)
    upper = np.array(
        [
            max(counts.max() * 2.0, 1.0),
            values.max(),
            max(values.std() * 5.0, 1.0),
            max(counts.max() * 2.0, 1.0),
            values.max(),
            max(values.std() * 5.0, 1.0),
        ],
        dtype=np.float64,
    )
    return p0, (lower, upper)


def sort_gaussians(params: np.ndarray) -> Tuple[GaussianParams, GaussianParams]:
    first = GaussianParams(float(params[0]), float(params[1]), float(params[2]))
    second = GaussianParams(float(params[3]), float(params[4]), float(params[5]))
    if first.mu <= second.mu:
        return first, second
    return second, first


def ashman_d(left: GaussianParams, right: GaussianParams) -> float:
    denom = np.sqrt(left.sd**2 + right.sd**2)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(2.0) * abs(left.mu - right.mu) / denom)


def surface_ratio(left: GaussianParams, right: GaussianParams) -> float:
    area1 = left.amplitude * left.sd * np.sqrt(2.0 * np.pi)
    area2 = right.amplitude * right.sd * np.sqrt(2.0 * np.pi)
    if area1 <= 0 or area2 <= 0:
        return 0.0
    return float(min(area1, area2) / max(area1, area2))


def bhattacharyya_coefficient(hist_norm: np.ndarray, fit_norm: np.ndarray) -> float:
    return float(np.sum(np.sqrt(np.clip(hist_norm, 0.0, None) * np.clip(fit_norm, 0.0, None))))


def fit_double_gaussian(
    values: np.ndarray,
    bins: int,
    quantile_clip: Optional[Tuple[float, float]] = None,
) -> DoubleGaussianFitResult:
    values = values[np.isfinite(values)]
    if values.size < 8:
        return _failed_fit("有效像元过少")

    hist = compute_histogram(values, bins=bins, quantile_clip=quantile_clip)
    counts = hist["counts"]
    centers = hist["centers"]
    try:
        otsu_value = otsu_threshold(values, bins=bins)
        p0, bounds = _prepare_initial_guess(values, centers, counts, otsu_value)
        params, _ = curve_fit(
            double_gaussian,
            centers,
            counts,
            p0=p0,
            bounds=bounds,
            maxfev=10000,
        )
        left, right = sort_gaussians(params)
        fit_curve = double_gaussian(
            centers,
            left.amplitude,
            left.mu,
            left.sd,
            right.amplitude,
            right.mu,
            right.sd,
        )
        fit_norm = fit_curve / fit_curve.sum() if fit_curve.sum() > 0 else np.zeros_like(fit_curve)
        ad = ashman_d(left, right)
        sr = surface_ratio(left, right)
        bc = bhattacharyya_coefficient(hist["norm"], fit_norm)
        return DoubleGaussianFitResult(
            success=True,
            left=left,
            right=right,
            ashman_d=ad,
            bc=bc,
            surface_ratio=sr,
            hist_range=(float(hist["range"][0]), float(hist["range"][1])),
            bins=bins,
            centers=centers,
            hist_counts=counts,
            hist_norm=hist["norm"],
            fit_curve=fit_curve,
            fit_norm=fit_norm,
            otsu_init=float(otsu_value),
        )
    except Exception as exc:  # noqa: BLE001
        return _failed_fit(str(exc))


def gaussian_intersection(left: GaussianParams, right: GaussianParams) -> Optional[float]:
    """求两个高斯曲线交点，返回位于两个均值之间的实根。"""

    a = 1.0 / (2 * left.sd**2) - 1.0 / (2 * right.sd**2)
    b = right.mu / (right.sd**2) - left.mu / (left.sd**2)
    c = (
        left.mu**2 / (2 * left.sd**2)
        - right.mu**2 / (2 * right.sd**2)
        + np.log((right.amplitude * left.sd) / (left.amplitude * right.sd))
    )
    coeffs = [a, b, c]
    roots = np.roots(coeffs)
    real_roots = [float(r.real) for r in roots if np.isreal(r)]
    lo, hi = sorted([left.mu, right.mu])
    inside = [r for r in real_roots if lo <= r <= hi]
    if inside:
        return float(inside[0])
    if real_roots:
        return float(sorted(real_roots, key=lambda r: abs(r - (lo + hi) * 0.5))[0])
    return None


def _failed_fit(message: str) -> DoubleGaussianFitResult:
    zeros = np.zeros(1, dtype=np.float64)
    empty = GaussianParams(0.0, 0.0, 1.0)
    return DoubleGaussianFitResult(
        success=False,
        left=empty,
        right=empty,
        ashman_d=0.0,
        bc=0.0,
        surface_ratio=0.0,
        hist_range=(0.0, 0.0),
        bins=0,
        centers=zeros,
        hist_counts=zeros,
        hist_norm=zeros,
        fit_curve=zeros,
        fit_norm=zeros,
        otsu_init=0.0,
        message=message,
    )
