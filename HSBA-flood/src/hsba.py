"""HSBA：分层四叉树寻找可双峰建模区域。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np

from .gaussian_fit import DoubleGaussianFitResult, fit_double_gaussian
from .hist_utils import extract_valid_values
from .quadtree import Tile


class TileSource(Protocol):
    """统一 XF 与差值 XC 的 tile 读取接口。"""

    @property
    def shape(self) -> Tuple[int, int]:
        ...

    def read_window(self, row_start: int, row_end: int, col_start: int, col_end: int) -> np.ndarray:
        ...


@dataclass(frozen=True)
class ArrayTileSource:
    array: np.ndarray

    @property
    def shape(self) -> Tuple[int, int]:
        return self.array.shape

    def read_window(self, row_start: int, row_end: int, col_start: int, col_end: int) -> np.ndarray:
        return self.array[row_start:row_end, col_start:col_end]


@dataclass(frozen=True)
class DifferenceTileSource:
    left: np.ndarray
    right: np.ndarray

    @property
    def shape(self) -> Tuple[int, int]:
        return self.left.shape

    def read_window(self, row_start: int, row_end: int, col_start: int, col_end: int) -> np.ndarray:
        return self.left[row_start:row_end, col_start:col_end] - self.right[row_start:row_end, col_start:col_end]


@dataclass
class TileEvaluation:
    tile: Tile
    valid_pixel_count: int
    valid_ratio: float
    fit: Optional[DoubleGaussianFitResult]
    selected: bool
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "tile": asdict(self.tile),
            "valid_pixel_count": self.valid_pixel_count,
            "valid_ratio": self.valid_ratio,
            "selected": self.selected,
            "reason": self.reason,
        }
        if self.fit is not None:
            result["fit"] = self.fit.to_dict()
        return result


def _build_integral_mask(valid_mask: np.ndarray) -> np.ndarray:
    """积分图用于 O(1) 统计 tile 内有效像元个数。"""

    return valid_mask.astype(np.int64).cumsum(axis=0, dtype=np.int64).cumsum(axis=1, dtype=np.int64)


def _window_sum(integral: np.ndarray, row_start: int, row_end: int, col_start: int, col_end: int) -> int:
    total = integral[row_end - 1, col_end - 1]
    if row_start > 0:
        total -= integral[row_start - 1, col_end - 1]
    if col_start > 0:
        total -= integral[row_end - 1, col_start - 1]
    if row_start > 0 and col_start > 0:
        total += integral[row_start - 1, col_start - 1]
    return int(total)


def run_hsba(
    source: TileSource,
    valid_mask: np.ndarray,
    config: Dict[str, Any],
    logger,
    image_name: str,
) -> Dict[str, Any]:
    """在单幅图像上运行 HSBA。"""

    selected_mask = np.zeros(source.shape, dtype=bool)
    level_stats: Dict[int, Dict[str, int]] = defaultdict(lambda: {"visited": 0, "selected": 0, "skipped": 0})
    selected_tiles: List[Dict[str, Any]] = []
    quantile_clip = tuple(config["histogram_quantile_clip"]) if config.get("histogram_quantile_clip") else None
    save_tile_details = bool(config.get("save_selected_tile_details", False))

    min_tile_size = int(config["min_tile_size"])
    min_valid_pixels = int(config["min_valid_pixels_per_tile"])
    min_valid_ratio = float(config["min_valid_ratio_per_tile"])
    std_epsilon = float(config.get("std_epsilon", 0.05))
    valid_integral = _build_integral_mask(valid_mask)

    def evaluate(tile: Tile) -> None:
        level_stats[tile.level]["visited"] += 1
        tile_area = tile.area
        valid_count = _window_sum(valid_integral, tile.row_start, tile.row_end, tile.col_start, tile.col_end)
        valid_ratio = valid_count / tile_area if tile_area > 0 else 0.0

        if valid_count < min_valid_pixels:
            level_stats[tile.level]["skipped"] += 1
            return
        if valid_ratio < min_valid_ratio:
            level_stats[tile.level]["skipped"] += 1
            return

        tile_mask = valid_mask[tile.row_start : tile.row_end, tile.col_start : tile.col_end]
        values = extract_valid_values(
            source.read_window(tile.row_start, tile.row_end, tile.col_start, tile.col_end),
            tile_mask,
        )
        if values.size < min_valid_pixels or float(values.std(ddof=0)) < std_epsilon:
            level_stats[tile.level]["skipped"] += 1
            return

        fit = fit_double_gaussian(values, bins=int(config["hist_bins"]), quantile_clip=quantile_clip)
        selected = (
            fit.success
            and fit.ashman_d > float(config["ashman_d_threshold"])
            and fit.bc > float(config["bc_threshold"])
            and fit.surface_ratio > float(config["sr_threshold"])
        )
        if save_tile_details:
            tile_eval = TileEvaluation(
                tile=tile,
                valid_pixel_count=valid_count,
                valid_ratio=valid_ratio,
                fit=fit if fit.success else None,
                selected=selected,
                reason="selected" if selected else ("fit_failed" if not fit.success else "criteria_not_met"),
            )

        if selected:
            selected_mask[tile.row_start : tile.row_end, tile.col_start : tile.col_end] |= tile_mask
            if save_tile_details:
                selected_tiles.append(tile_eval.to_dict())
            level_stats[tile.level]["selected"] += 1
            return

        children = tile.split()
        if not children:
            level_stats[tile.level]["skipped"] += 1
            return
        if tile.height <= min_tile_size or tile.width <= min_tile_size:
            level_stats[tile.level]["skipped"] += 1
            return
        for child in children:
            evaluate(child)

    root = Tile(level=0, row_start=0, row_end=source.shape[0], col_start=0, col_end=source.shape[1])
    evaluate(root)

    selected_tile_count = sum(stats["selected"] for stats in level_stats.values())

    logger.info("%s HSBA 结束: 选中 tile=%d, BM 像元=%d", image_name, selected_tile_count, int(selected_mask.sum()))
    for level in sorted(level_stats):
        stats = level_stats[level]
        logger.info(
            "%s level=%d visited=%d selected=%d skipped=%d",
            image_name,
            level,
            stats["visited"],
            stats["selected"],
            stats["skipped"],
        )

    return {
        "mask": selected_mask,
        "selected_tile_count": selected_tile_count,
        "selected_tiles": selected_tiles,
        "level_stats": {str(k): v for k, v in sorted(level_stats.items())},
    }


def run_hsba_with_fallback(
    source: TileSource,
    valid_mask: np.ndarray,
    config: Dict[str, Any],
    logger,
    image_name: str,
) -> Dict[str, Any]:
    """BM 太少时按约束做一次温和回退。"""

    result = run_hsba(source, valid_mask, config, logger, image_name)
    if int(result["mask"].sum()) >= int(config.get("bm_min_pixels", 4096)):
        return result
    if not bool(config.get("fallback_relaxations", True)):
        return result

    relaxed = dict(config)
    relaxed["min_valid_ratio_per_tile"] = max(0.3, float(config["min_valid_ratio_per_tile"]) - 0.15)
    relaxed["min_tile_size"] = max(64, int(config["min_tile_size"]) // 2)
    logger.warning(
        "%s BM 像元不足(%d)，尝试回退: min_valid_ratio_per_tile=%.3f, min_tile_size=%d",
        image_name,
        int(result["mask"].sum()),
        float(relaxed["min_valid_ratio_per_tile"]),
        int(relaxed["min_tile_size"]),
    )
    fallback = run_hsba(source, valid_mask, relaxed, logger, f"{image_name}_fallback")
    if int(fallback["mask"].sum()) > int(result["mask"].sum()):
        fallback["fallback_used"] = True
        fallback["fallback_config"] = {
            "min_valid_ratio_per_tile": float(relaxed["min_valid_ratio_per_tile"]),
            "min_tile_size": int(relaxed["min_tile_size"]),
        }
        return fallback
    result["fallback_used"] = False
    return result
