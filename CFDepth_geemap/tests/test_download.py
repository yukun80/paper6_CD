import tempfile
import unittest
from pathlib import Path
import numpy as np
import rasterio
from affine import Affine
from cfdepth.download import validate_raster


class DownloadTests(unittest.TestCase):
    def test_multiblock_grid_and_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "depth.tif"
            grid = {
                "crs": "EPSG:4326",
                "transform": [1 / 3600, 0, 110, 0, -1 / 3600, 30],
                "shape": [65, 67],
            }
            data = np.arange(65 * 67, dtype="float32").reshape(65, 67) + 1
            data[::7, ::3] = -9999
            profile = dict(
                driver="GTiff",
                width=67,
                height=65,
                count=1,
                dtype="float32",
                crs=grid["crs"],
                transform=Affine(*grid["transform"]),
                nodata=-9999,
                tiled=True,
                blockxsize=16,
                blockysize=16,
            )
            with rasterio.open(path, "w", **profile) as dst:
                for _, window in dst.block_windows(1):
                    r, c = int(window.row_off), int(window.col_off)
                    dst.write(
                        data[r : r + int(window.height), c : c + int(window.width)],
                        1,
                        window=window,
                    )
            stats = validate_raster(path, grid, -9999, "depth", 0.01)
            self.assertEqual(stats["valid_pixels"], int((data != -9999).sum()))
            self.assertEqual(
                stats["sum"], float(data[data != -9999].sum(dtype=np.float64))
            )
            wrong = dict(grid, shape=[66, 67])
            with self.assertRaises(ValueError):
                validate_raster(path, wrong, -9999, "depth", 0.01)
            with rasterio.open(path, "r+") as dst:
                dst.nodata = -1
            with self.assertRaises(ValueError):
                validate_raster(path, grid, -9999, "depth", 0.01)


if __name__ == "__main__":
    unittest.main()


class TransportTests(unittest.TestCase):
    def test_scoped_proxy_session_restored(self):
        from geedim.utils import AsyncRunner
        from cfdepth.download import download_transport

        runner = AsyncRunner()
        descriptor = type(runner).session
        with self.assertRaisesRegex(RuntimeError, "simulated"):
            with download_transport():
                session = runner.session
                self.assertTrue(session.trust_env)
                raise RuntimeError("simulated")
        self.assertTrue(session.closed)
        self.assertIs(type(runner).session, descriptor)
