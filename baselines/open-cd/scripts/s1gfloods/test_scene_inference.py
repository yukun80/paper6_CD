"""单切片推理兼容与输出路径回归检查，不加载模型权重。"""
import argparse
import json
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
import rasterio
from rasterio.transform import from_origin

SOURCE = Path(__file__).resolve().parents[2] / 'tools/infer_gf3_henan.py'
spec = importlib.util.spec_from_file_location('scene_infer', SOURCE)
infer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(infer)


class FakeInferencer:
    def __init__(self):
        self.calls = []

    def preprocess(self, inputs, batch_size):
        assert batch_size == 1
        for item in inputs:
            yield [item]

    def forward(self, inputs):
        assert len(inputs) == 1
        self.calls.append(inputs[0])
        return inputs


class SceneInferenceTests(unittest.TestCase):
    def test_four_then_tail_preserve_order(self):
        model = FakeInferencer()
        result = []
        for group in ([0, 1, 2, 3], [4]):
            result.extend(infer.predict_single_tiles(model, group))
        self.assertEqual(result, list(range(5)))
        self.assertEqual(model.calls, list(range(5)))

    def test_missing_prediction_rejected(self):
        model = FakeInferencer()
        model.forward = lambda inputs: []
        with self.assertRaisesRegex(RuntimeError, 'one prediction'):
            infer.predict_single_tiles(model, [1])

    def test_missing_preprocessed_input_rejected(self):
        model = FakeInferencer()
        model.preprocess = lambda inputs, batch_size: iter([[inputs[0]]])
        with self.assertRaisesRegex(RuntimeError, 'count mismatch'):
            infer.predict_single_tiles(model, [1, 2])

    def test_paths_and_explicit_override(self):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(output_root=directory, config='bit.py',
                                      data_root='', out_dir='', mosaic_dir='', skip_mosaic=False)
            for source, region in [('GF3_Henan_CD_infer', 'Zhengzhou'),
                                   ('GF3_Zhuozhou_CD_infer', 'Zhuozhou'),
                                   ('LT1_Guangxi_CD_infer', 'Guangxi')]:
                args.data_root = source
                root = Path(directory) / 'bit' / region
                self.assertEqual(infer.ensure_tile_output_dir(args), root / 'tile_png')
                self.assertEqual(infer.ensure_mosaic_output_dir(args), root / 'mosaic')
            args.out_dir = str(Path(directory) / 'custom_tiles')
            args.mosaic_dir = str(Path(directory) / 'custom_mosaic')
            self.assertEqual(infer.ensure_tile_output_dir(args), Path(args.out_dir))
            self.assertEqual(infer.ensure_mosaic_output_dir(args), Path(args.mosaic_dir))
            args.skip_mosaic = True
            self.assertIsNone(infer.ensure_mosaic_output_dir(args))

    def test_mosaic_nodata_and_partial_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.tif'
            transform = from_origin(100, 200, 10, 10)
            with rasterio.open(source, 'w', driver='GTiff', width=2, height=2,
                               count=1, dtype='uint8', crs='EPSG:32649',
                               transform=transform) as dst:
                dst.write(np.zeros((2, 2), dtype='uint8'), 1)
            report = {'source': {'pre_image': str(source), 'height': 2, 'width': 2}}
            infer.finalize_mosaic_outputs(
                report, root, np.array([[0.8, 0.2], [0, 1.2]], dtype='float32'),
                np.array([[1, 1], [0, 2]], dtype='float32'), 0.5,
                data_root=root, config_path=root / 'config.py', checkpoint_path=root / 'model.pth',
                tile_output_dir=root / 'tile_png', total_tiles=10, used_tiles=2)
            with rasterio.open(root / 'change_binary.tif') as ds:
                np.testing.assert_array_equal(ds.read(1), [[1, 0], [255, 1]])
                self.assertEqual(ds.nodata, 255)
                self.assertEqual(ds.transform, transform)
            with rasterio.open(root / 'change_prob.tif') as ds:
                self.assertEqual(ds.nodata, -1)
                np.testing.assert_allclose(ds.read(1), [[0.8, 0.2], [-1, 0.6]])
            self.assertTrue((root / 'change_binary.png').is_file())
            self.assertTrue(json.loads((root / 'infer_report.json').read_text())['partial_run'])


if __name__ == '__main__':
    unittest.main()
