"""真实9块下载验收：每像元唯一值、NoData孔洞和原网格，不创建资产。"""

import os
from pathlib import Path
import tempfile
import numpy as np
import rasterio
import ee
from cfdepth.config import Config
from cfdepth.components import inputs
from cfdepth.grid import Grid
from cfdepth.download import download_transport, validate_raster
from cfdepth.storage import atomic_json


def main():
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path("CFDepth_geemap/.cache/matplotlib").resolve())
    )
    from geemap.common import download_ee_image

    cfg = Config.read("CFDepth_geemap/configs/small.json")
    ee.Initialize(project=cfg.project)
    _, _, grid, region = inputs(cfg)
    g = Grid(grid, region, cfg.p)
    x, y = g.xy.select("x"), g.xy.select("y")
    image = (
        x.add(y.multiply(100))
        .add(0.25)
        .toFloat()
        .clip(region)
        .updateMask(x.mod(7).neq(0))
        .unmask(-9999, False)
    )
    with tempfile.TemporaryDirectory(prefix="cfdepth-download-") as directory:
        path = Path(directory) / "tiled.tif"
        with download_transport():
            download_ee_image(
                image,
                str(path),
                crs=grid["crs"],
                crs_transform=grid["transform"],
                shape=[36, 36],
                dtype="float32",
                resampling="near",
                nodata=-9999,
                max_tile_dim=16,
                max_tile_size=1,
                max_requests=2,
                max_cpus=2,
            )
        row, col = np.indices((36, 36))
        expected = (col + 100 * row + 0.25).astype("float32")
        expected[col % 7 == 0] = -9999
        with rasterio.open(path) as src:
            actual = src.read(1)
        assert np.array_equal(
            actual, expected
        ), "Tile gap, duplicate, seam, value or mask mismatch"
        report = validate_raster(path, dict(grid, shape=[36, 36]), -9999, "depth", 0.01)
    atomic_json(
        Path("CFDepth_geemap/environment/cloud-download-probe.json"),
        {"passed": True, "tiles": 9, "all_1296_pixels_equal": True, "report": report},
    )
    print("PASS: 9 tiles, all 1296 pixels including NoData match independent indices")


if __name__ == "__main__":
    main()
