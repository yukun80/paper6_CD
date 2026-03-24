"""预处理：读取影像、构造有效掩膜与变化图。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from .io_utils import align_raster_to_reference, build_run_metadata, grids_match, read_raster
from .nodata_utils import build_valid_mask, combine_valid_masks, count_valid_ratio


def prepare_input_pair(config: Dict[str, Any], config_dir: Path, logger) -> Dict[str, Any]:
    """读取并对齐 XR/XF，只保留低内存主流程所需数组。"""

    xr = read_raster((config_dir / config["pre_path"]).resolve())
    xf = read_raster((config_dir / config["post_path"]).resolve())

    align_to_grid = str(config.get("align_to_grid", "post")).lower()
    if not grids_match(xr, xf):
        logger.warning("输入影像网格不一致，开始按配置对齐: align_to_grid=%s", align_to_grid)
        if align_to_grid == "post":
            xr = align_raster_to_reference(xr, xf)
        elif align_to_grid == "pre":
            xf = align_raster_to_reference(xf, xr)
        else:
            raise ValueError(f"不支持的 align_to_grid: {align_to_grid}")

    nodata_override = config.get("nodata_override")
    valid_xr = build_valid_mask(xr["array"], xr["nodata"], nodata_override)
    valid_xf = build_valid_mask(xf["array"], xf["nodata"], nodata_override)
    valid_mask = combine_valid_masks(valid_xr, valid_xf)

    valid_pixels, total_pixels, ratio = count_valid_ratio(valid_mask)
    logger.info("有效像元: %d / %d (%.4f)", valid_pixels, total_pixels, ratio)

    return {
        "XR": xr["array"].astype(np.float32),
        "XF": xf["array"].astype(np.float32),
        "valid_mask": valid_mask,
        "profile": xf["profile"] if align_to_grid == "post" or grids_match(xr, xf) else xr["profile"],
        "transform": xf["transform"] if align_to_grid == "post" or grids_match(xr, xf) else xr["transform"],
        "crs": xf["crs"] if align_to_grid == "post" or grids_match(xr, xf) else xr["crs"],
        "xr_meta": build_run_metadata(xr),
        "xf_meta": build_run_metadata(xf),
        "valid_stats": {
            "valid_pixels": valid_pixels,
            "total_pixels": total_pixels,
            "valid_ratio": ratio,
        },
    }
