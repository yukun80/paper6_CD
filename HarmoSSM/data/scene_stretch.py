"""独立的 SAR 整景双时相拉伸工具；与 scene_pair_histogram_v1 保持数值一致。

只拟合整景标尺和应用冻结映射，不负责切片划分、标签处理或模型加载。
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import rasterio
from rasterio.windows import Window


STRETCH_ALGORITHM_VERSION = "scene_pair_histogram_v1"
HISTOGRAM_BINS = 65536
SCAN_BLOCK_SIZE = 1024


@dataclass(frozen=True)
class StretchConfig:
    """整景双时相共用的灰度拉伸拟合配置。"""

    mode: str = "percentile"
    low: float = 2.0
    high: float = 98.0
    value_min: float | None = None
    value_max: float | None = None


@dataclass(frozen=True)
class StretchBounds:
    """已拟合且不随切片变化的映射上下限。"""

    low: float
    high: float


def validate_stretch(config: StretchConfig) -> None:
    if config.mode not in {"percentile", "value"}:
        raise ValueError("stretch_mode must be 'percentile' or 'value'")
    if config.mode == "percentile":
        if not 0.0 <= config.low < config.high <= 100.0:
            raise ValueError("Percentile stretch requires 0 <= low < high <= 100")
        return
    if config.value_min is None or config.value_max is None:
        raise ValueError("Value stretch requires value_min and value_max")
    if not np.isfinite(config.value_min) or not np.isfinite(config.value_max):
        raise ValueError("Value stretch bounds must be finite")
    if config.value_min >= config.value_max:
        raise ValueError("value_min must be smaller than value_max")


def build_valid_mask(
    pre: np.ndarray,
    pre_nodata: float | int | None,
    post: np.ndarray,
    post_nodata: float | int | None,
) -> np.ndarray:
    """灾前和灾后同时为有限且非 NoData 时才视为有效。"""
    valid = np.isfinite(pre) & np.isfinite(post)
    if pre_nodata is not None:
        valid &= np.not_equal(pre, pre_nodata)
    if post_nodata is not None:
        valid &= np.not_equal(post, post_nodata)
    return valid


def stretch_to_uint8(
    values: np.ndarray,
    valid_mask: np.ndarray,
    bounds: StretchBounds,
) -> np.ndarray:
    """只应用冻结的上下限；禁止在窗口内重新估计统计量。"""
    output = np.zeros(values.shape, dtype=np.uint8)
    if not np.any(valid_mask) or bounds.high == bounds.low:
        return output
    scaled = np.clip(
        (values[valid_mask].astype(np.float64) - bounds.low) / (bounds.high - bounds.low),
        0.0,
        1.0,
    )
    output[valid_mask] = np.rint(scaled * 255.0).astype(np.uint8)
    return output


def scan_windows(width: int, height: int, block_size: int = SCAN_BLOCK_SIZE) -> Iterator[Window]:
    """按原图非重叠块扫描，避免滑窗重叠造成重复计数。"""
    if block_size < 1:
        raise ValueError("scan block size must be positive")
    for top in range(0, height, block_size):
        for left in range(0, width, block_size):
            yield Window(left, top, min(block_size, width - left), min(block_size, height - top))


def read_scene_window(
    pre_ds: rasterio.io.DatasetReader,
    post_ds: rasterio.io.DatasetReader,
    window: Window,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """统计和切片共用原始数值及掩膜，包含 GDAL 内部/外部有效掩膜。"""
    pre = pre_ds.read(1, window=window)
    post = post_ds.read(1, window=window)
    valid = build_valid_mask(pre, pre_ds.nodata, post, post_ds.nodata)
    valid &= pre_ds.read_masks(1, window=window) > 0
    valid &= post_ds.read_masks(1, window=window) > 0
    return pre, post, valid


def histogram_percentile(histogram: np.ndarray, minimum: float, maximum: float, q: float) -> float:
    """按零基排序位置，以箱中心估计次序值并线性插值。"""
    cumulative = np.cumsum(histogram, dtype=np.int64)
    count = int(cumulative[-1])
    if count == 0 or not 0 <= q <= 100:
        raise ValueError("Histogram must be nonempty and percentile in [0, 100]")
    if q == 0 or minimum == maximum:
        return minimum
    if q == 100:
        return maximum
    rank = (count - 1) * q / 100.0
    lower, upper = int(np.floor(rank)), int(np.ceil(rank))
    bins = np.searchsorted(cumulative, [lower, upper], side="right")
    centers = minimum + (bins.astype(np.float64) + 0.5) * ((maximum - minimum) / len(histogram))
    return float(centers[0] + (centers[1] - centers[0]) * (rank - lower))


def fit_scene_stretch(
    pre_ds: rasterio.io.DatasetReader,
    post_ds: rasterio.io.DatasetReader,
    config: StretchConfig,
    *,
    block_size: int = SCAN_BLOCK_SIZE,
) -> tuple[StretchBounds, dict[str, Any]]:
    """全景两遍拟合共同标尺，第三遍精确统计各时相截断比例。"""
    validate_stretch(config)
    minimum, maximum, count = float("inf"), float("-inf"), 0
    for window in scan_windows(pre_ds.width, pre_ds.height, block_size):
        pre, post, valid = read_scene_window(pre_ds, post_ds, window)
        count += int(valid.sum())
        if valid.any():
            for values in (pre[valid], post[valid]):
                minimum = min(minimum, float(values.min()))
                maximum = max(maximum, float(values.max()))
    if count == 0:
        raise ValueError("Scene pair has no common valid pixels; no output written")
    if not np.isfinite(maximum - minimum):
        raise ValueError("Scene value range cannot be represented in float64")

    bin_width = 0.0
    fallback = None
    if config.mode == "value":
        bounds = StretchBounds(float(config.value_min), float(config.value_max))
    elif minimum == maximum:
        bounds = StretchBounds(minimum, maximum)
        fallback = "constant_scene_zero_output"
    else:
        edges = np.linspace(minimum, maximum, HISTOGRAM_BINS + 1, dtype=np.float64)
        histogram = np.zeros(HISTOGRAM_BINS, dtype=np.int64)
        for window in scan_windows(pre_ds.width, pre_ds.height, block_size):
            pre, post, valid = read_scene_window(pre_ds, post_ds, window)
            for values in (pre[valid], post[valid]):
                histogram += np.histogram(values.astype(np.float64), bins=edges)[0]
        bin_width = float(np.max(np.diff(edges)))
        bounds = StretchBounds(
            histogram_percentile(histogram, minimum, maximum, config.low),
            histogram_percentile(histogram, minimum, maximum, config.high),
        )
        if bounds.high <= bounds.low:
            bounds = StretchBounds(minimum, maximum)
            fallback = "joint_min_max"

    clipping = {phase: {"below_low": 0, "above_high": 0} for phase in ("pre", "post")}
    for window in scan_windows(pre_ds.width, pre_ds.height, block_size):
        pre, post, valid = read_scene_window(pre_ds, post_ds, window)
        for phase, values in (("pre", pre[valid]), ("post", post[valid])):
            # 与映射算术同用 float64，避免 NumPy 将阈值降为 float32 后改变边界计数。
            values = values.astype(np.float64)
            clipping[phase]["below_low"] += int(np.count_nonzero(values < bounds.low))
            clipping[phase]["above_high"] += int(np.count_nonzero(values > bounds.high))
    for counts in clipping.values():
        counts["below_low_ratio"] = counts["below_low"] / count
        counts["above_high_ratio"] = counts["above_high"] / count

    return bounds, {
        "algorithm_version": STRETCH_ALGORITHM_VERSION,
        "policy": "scene_pair_adaptive" if config.mode == "percentile" else "fixed_value",
        "scope": "full_scene_common_valid_pixels_both_times_equal_weight",
        "value_domain": "as_provided_no_log_or_histogram_matching",
        "mask_policy": "finite_and_declared_nodata_and_both_raster_masks",
        "bounds": asdict(bounds),
        "joint_min": minimum,
        "joint_max": maximum,
        "common_valid_pixels": count,
        "joint_value_count": 2 * count,
        "histogram_bins": HISTOGRAM_BINS if config.mode == "percentile" else 0,
        "histogram_bin_width": bin_width,
        "percentile_absolute_error_bound": bin_width,
        "percentile_estimator": "linear_rank_bin_centers" if config.mode == "percentile" else None,
        "fallback": fallback,
        "scan_block_size": block_size,
        "clipping": clipping,
        "compatibility_note": "Changed input distribution for existing checkpoints; accuracy not validated.",
    }
