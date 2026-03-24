"""科研复核用图件输出。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .gaussian_fit import DoubleGaussianFitResult, gaussian


def _resolve_stride(shape: tuple[int, int], max_pixels: int, stride: Optional[int]) -> int:
    if stride is not None and stride > 1:
        return int(stride)
    total_pixels = int(shape[0] * shape[1])
    if total_pixels <= max_pixels:
        return 1
    return int(np.ceil(np.sqrt(total_pixels / max_pixels)))


def _downsample(array: np.ndarray, stride: int) -> np.ndarray:
    return array[::stride, ::stride] if stride > 1 else array


def _masked_image(array: np.ndarray, valid_mask: np.ndarray) -> np.ma.MaskedArray:
    return np.ma.array(array, mask=~valid_mask)


def save_float_preview(
    path: str | Path,
    array: np.ndarray,
    valid_mask: np.ndarray,
    title: str,
    cmap: str = "gray",
    max_pixels: int = 2_000_000,
    stride: Optional[int] = None,
) -> None:
    stride = _resolve_stride(array.shape, max_pixels=max_pixels, stride=stride)
    fig, ax = plt.subplots(figsize=(10, 6))
    image = _masked_image(_downsample(array, stride), _downsample(valid_mask, stride))
    im = ax.imshow(image, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_difference_preview(
    path: str | Path,
    left: np.ndarray,
    right: np.ndarray,
    valid_mask: np.ndarray,
    title: str,
    cmap: str = "coolwarm",
    max_pixels: int = 2_000_000,
    stride: Optional[int] = None,
) -> None:
    stride = _resolve_stride(left.shape, max_pixels=max_pixels, stride=stride)
    diff = _downsample(left, stride) - _downsample(right, stride)
    downsampled_valid = _downsample(valid_mask, stride)
    fig, ax = plt.subplots(figsize=(10, 6))
    image = _masked_image(diff, downsampled_valid)
    im = ax.imshow(image, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_mask_preview(
    path: str | Path,
    mask: np.ndarray,
    valid_mask: np.ndarray,
    title: str,
    max_pixels: int = 2_000_000,
    stride: Optional[int] = None,
) -> None:
    stride = _resolve_stride(mask.shape, max_pixels=max_pixels, stride=stride)
    downsampled_mask = _downsample(mask, stride)
    downsampled_valid = _downsample(valid_mask, stride)
    fig, ax = plt.subplots(figsize=(10, 6))
    background = np.zeros_like(downsampled_mask, dtype=np.float32)
    background[downsampled_valid] = downsampled_mask[downsampled_valid].astype(np.float32)
    image = np.ma.array(background, mask=~downsampled_valid)
    ax.imshow(image, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_hist_fit_plot(
    path: str | Path,
    fit_result: DoubleGaussianFitResult,
    title: str,
    left_label: str,
    right_label: str,
    extra_lines: Optional[Dict[str, Optional[float]]] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    centers = fit_result.centers
    bar_width = centers[1] - centers[0] if centers.size > 1 else 1.0
    ax.bar(centers, fit_result.hist_norm, width=bar_width, alpha=0.4, label="Empirical PDF")
    left_curve = gaussian(centers, fit_result.left.amplitude, fit_result.left.mu, fit_result.left.sd)
    right_curve = gaussian(centers, fit_result.right.amplitude, fit_result.right.mu, fit_result.right.sd)
    total = fit_result.fit_curve
    if total.sum() > 0:
        ax.plot(centers, left_curve / total.sum(), label=left_label, linewidth=2)
        ax.plot(centers, right_curve / total.sum(), label=right_label, linewidth=2)
        ax.plot(centers, fit_result.fit_norm, label="Double Gaussian", linewidth=2, linestyle="--")
    ax.axvline(fit_result.left.mu, color="tab:blue", linestyle=":", label="mu_left")
    ax.axvline(fit_result.right.mu, color="tab:orange", linestyle=":", label="mu_right")
    if extra_lines:
        for label, value in extra_lines.items():
            if value is not None:
                ax.axvline(value, linestyle="-.", linewidth=1.5, label=label)
    ax.set_title(title)
    ax.set_xlabel("dB")
    ax.set_ylabel("Normalized Frequency")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_rmse_heatmap(
    path: str | Path,
    rmse_grid: np.ndarray,
    sigma_rg_grid: np.ndarray,
    delta_cd_grid: np.ndarray,
    sigma_rg_best: float,
    delta_sigma_cd_best: float,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(rmse_grid, aspect="auto", origin="lower", cmap="magma")
    ax.set_title("RMSE Search Heatmap")
    ax.set_xlabel("delta_sigma_CD")
    ax.set_ylabel("sigma_RG")
    x_ticks = np.linspace(0, len(delta_cd_grid) - 1, num=min(8, len(delta_cd_grid)), dtype=int)
    y_ticks = np.linspace(0, len(sigma_rg_grid) - 1, num=min(8, len(sigma_rg_grid)), dtype=int)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{delta_cd_grid[i]:.1f}" for i in x_ticks])
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{sigma_rg_grid[i]:.1f}" for i in y_ticks])
    best_y = int(np.argmin(np.abs(sigma_rg_grid - sigma_rg_best)))
    best_x = int(np.argmin(np.abs(delta_cd_grid - delta_sigma_cd_best)))
    ax.scatter([best_x], [best_y], c="cyan", s=50, marker="x")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_flood_overlay(
    path: str | Path,
    xf: np.ndarray,
    valid_mask: np.ndarray,
    flood_mask: np.ndarray,
    max_pixels: int = 2_000_000,
    stride: Optional[int] = None,
) -> None:
    stride = _resolve_stride(xf.shape, max_pixels=max_pixels, stride=stride)
    fig, ax = plt.subplots(figsize=(10, 6))
    base = _masked_image(_downsample(xf, stride), _downsample(valid_mask, stride))
    ax.imshow(base, cmap="gray")
    downsampled_flood = _downsample(flood_mask, stride)
    overlay = np.ma.array(downsampled_flood.astype(np.float32), mask=~downsampled_flood.astype(bool))
    ax.imshow(overlay, cmap="autumn", alpha=0.5, vmin=0.0, vmax=1.0)
    ax.set_title("Final Flood Overlay on XF")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
