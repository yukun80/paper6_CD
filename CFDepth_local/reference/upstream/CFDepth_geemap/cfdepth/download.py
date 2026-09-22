"""固定网格分块直传本地；核验完成才发布两个产品。"""

from contextlib import contextmanager
import json
import logging
import math
import os
from pathlib import Path
from unittest.mock import patch
import numpy as np
import rasterio
import ee
from .grid import Grid
from .storage import atomic_json, sha256


@contextmanager
def download_transport():
    """geedim 2.0未开放session配置，仅在本次下载继承环境代理。"""
    import aiohttp
    from geedim.utils import AsyncRunner

    runner = AsyncRunner()

    async def create():
        return aiohttp.ClientSession(
            trust_env=True,
            raise_for_status=True,
            timeout=aiohttp.ClientTimeout(total=180, sock_connect=30),
        )

    session = runner.run(create())
    try:
        with patch.object(type(runner), "session", property(lambda self: session)):
            yield
    finally:
        runner.run(session.close())


def output_grid(grid: dict, region: ee.Geometry) -> dict:
    points = region.bounds(1, ee.Projection(grid["crs"])).coordinates().getInfo()[0]
    t = grid["transform"]
    xs = [(p[0] - t[2]) / t[0] for p in points]
    ys = [(p[1] - t[5]) / t[4] for p in points]

    def snap(v):
        return round(v) if abs(v - round(v)) < 1e-6 else v

    col, row = math.floor(snap(min(xs))), math.floor(snap(min(ys)))
    width, height = math.ceil(snap(max(xs))) - col, math.ceil(snap(max(ys))) - row
    return {
        "crs": grid["crs"],
        "transform": [t[0], 0, t[2] + col * t[0], 0, t[4], t[5] + row * t[4]],
        "shape": [height, width],
    }


def validate_raster(
    path: str | Path, grid: dict, nodata: float, kind: str, min_depth: float
) -> dict:
    total = 0
    low = None
    high = None
    summation = 0.0
    with rasterio.open(path) as src:
        if src.count != 1 or src.dtypes != ("float32",) or src.nodata != nodata:
            raise ValueError("Unexpected output band/type/NoData: " + str(path))
        if (
            src.crs != rasterio.crs.CRS.from_user_input(grid["crs"])
            or [src.height, src.width] != grid["shape"]
        ):
            raise ValueError("Output CRS/dimensions changed")
        if not np.allclose(
            tuple(src.transform)[:6],
            grid["transform"],
            rtol=0,
            atol=abs(grid["transform"][0]) * 1e-6,
        ):
            raise ValueError("Output affine changed")
        for _, window in src.block_windows(1):
            a = src.read(1, window=window)
            valid = src.read_masks(1, window=window) > 0
            if not np.array_equal(valid, a != nodata):
                raise ValueError("NoData/mask disagreement")
            v = a[valid]
            # 云端在FP64判定 > minDepth；Float32保存可能恰好舍入为该阈值。
            if (
                not np.isfinite(v).all()
                or (kind == "depth" and np.any(v < np.float32(min_depth)))
                or (kind == "gradient" and np.any(v < 0))
            ):
                raise ValueError("Invalid output values")
            if v.size:
                lo, hi = float(v.min()), float(v.max())
                low = lo if low is None else min(low, lo)
                high = hi if high is None else max(high, hi)
                total += v.size
                summation += float(v.sum(dtype=np.float64))
    return {
        "valid_pixels": int(total),
        "min": low,
        "max": high,
        "sum": summation,
        "sha256": sha256(path),
    }


def download_products(backend, journal, save):
    from geemap.common import download_ee_image

    cfg = backend.cfg
    folder = cfg.output_dir
    folder.mkdir(parents=True, exist_ok=True)
    owner = folder / "owner.json"
    identity = {"identity": cfg.identity, "run_id": cfg.run_id}
    if owner.exists():
        if json.loads(owner.read_text()) != identity:
            raise ValueError("Output directory belongs to another run")
    else:
        if any(folder.iterdir()):
            raise ValueError("Refusing nonempty unowned output directory")
        atomic_json(owner, identity)
    images, audit = backend.products(journal)
    logging.getLogger("cfdepth").info("Final component audit: %s", audit)
    grid = output_grid(backend.grid, backend.region)
    g = Grid(backend.grid, backend.region, cfg.p)
    counts = g.total(
        g.pack([g.indicator(im.mask().gt(0)).rename(k) for k, im in images.items()]),
        ee.Reducer.sum().unweighted(),
    ).getInfo()
    journal["final_audit"] = audit
    journal["output_grid"] = grid
    journal.setdefault("downloads", {})
    save()
    reports = {}
    for kind, image in images.items():
        target = folder / (
            "CFDepth.tif" if kind == "depth" else "CFDepth_WSE_gradient.tif"
        )
        tmp = folder / (target.stem + ".partial.tif")
        old = journal["downloads"].get(kind)
        if target.exists():
            if not old or old.get("sha256") != sha256(target):
                raise ValueError(
                    "Existing output has no matching receipt: " + str(target)
                )
            report = validate_raster(target, grid, cfg.p.noData, kind, cfg.p.minDepth)
        else:
            with download_transport():
                download_ee_image(
                    image.toFloat().unmask(cfg.p.noData, False),
                    str(tmp),
                    crs=grid["crs"],
                    crs_transform=grid["transform"],
                    shape=grid["shape"],
                    dtype="float32",
                    resampling="near",
                    max_requests=cfg.data["download_threads"],
                    max_cpus=cfg.data["download_threads"],
                    max_tile_size=16,
                    max_tile_dim=2048,
                    nodata=cfg.p.noData,
                    overwrite=True,
                )
            report = validate_raster(tmp, grid, cfg.p.noData, kind, cfg.p.minDepth)
        if report["valid_pixels"] != counts[kind]:
            raise ValueError("Download valid-pixel count mismatch")
        if cfg.fixture == "small":
            if kind == "depth" and (
                report["valid_pixels"] != 415
                or report["min"] is None
                or max(abs(report["min"] - 0.5), abs(report["max"] - 0.5)) > 1e-7
            ):
                raise ValueError("Small depth acceptance failed: " + str(report))
            if kind == "gradient" and (
                not report["valid_pixels"] or report["max"] > 1e-10
            ):
                raise ValueError("Small gradient acceptance failed")
        journal["downloads"][kind] = report
        save()
        if not target.exists():
            os.replace(tmp, target)
        reports[kind] = report
    with rasterio.open(folder / "CFDepth.tif") as depth, rasterio.open(
        folder / "CFDepth_WSE_gradient.tif"
    ) as grad:
        for _, window in depth.block_windows(1):
            if np.any(
                (grad.read_masks(1, window=window) > 0)
                & (depth.read_masks(1, window=window) == 0)
            ):
                raise ValueError("Gradient outside final water-depth support")
    atomic_json(
        folder / "report.json",
        {
            "identity": cfg.identity,
            "audit": audit,
            "grid": grid,
            "units": {"depth": "m", "gradient": "m/m"},
            "products": reports,
        },
    )
    return reports
