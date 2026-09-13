"""单切片推理兼容与输出路径回归检查，不加载模型权重。"""
import argparse
import json
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
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
    def test_probability_and_threshold_boundaries(self):
        from opencd.models.change_detectors.siamencoder_decoder import SiamEncoderDecoder
        torch = infer.torch
        for channels, threshold in [(1, 0.3), (1, 0.5), (2, 0.5)]:
            probability = torch.tensor([threshold - 0.1, threshold, threshold + 0.1])
            if channels == 1:
                logits = torch.logit(probability).reshape(1, 1, 1, 3)
            else:
                logits = torch.stack([torch.zeros(3), torch.tensor([-1., 0., 1.])]).reshape(1, 2, 1, 3)
            holder = SimpleNamespace(decode_head=SimpleNamespace(threshold=threshold))
            prediction = SiamEncoderDecoder.postprocess_result(holder, logits)[0]
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'tile.png'
                infer.save_binary_prediction(prediction, path)
                np.testing.assert_array_equal(np.asarray(infer.Image.open(path)).reshape(-1), [0, 0, 255])
                native = prediction.pred_sem_seg.data.squeeze().numpy().astype('uint8') * 255
                np.testing.assert_array_equal(np.asarray(infer.Image.open(path)).reshape(-1), native)

    def test_decision_rule_and_model_validation(self):
        for head, threshold, source in [
            (dict(out_channels=1), 0.3, 'mmseg_default'),
            (dict(out_channels=1, threshold=0.5), 0.5, 'training_config'),
            (dict(num_classes=2), 0.5, 'two_class_argmax'),
        ]:
            rule = infer.resolve_decision_rule(infer.Config(dict(model=dict(decode_head=head))))
            self.assertEqual((rule['threshold'], rule['threshold_source']), (threshold, source))
            infer.verify_decision_rule(SimpleNamespace(decode_head=SimpleNamespace(
                out_channels=rule['out_channels'], threshold=threshold)), rule)
            with self.assertRaises(ValueError):
                infer.verify_decision_rule(SimpleNamespace(decode_head=SimpleNamespace(out_channels=3)), rule)
        with self.assertRaises(ValueError):
            infer.resolve_decision_rule(infer.Config(dict(model=dict(decode_head=dict(out_channels=3)))))

    def test_removed_threshold_argument(self):
        import sys
        import subprocess
        with patch.object(sys, 'argv', ['infer', 'config.py', 'model.pth', '--threshold', '0.5']):
            with self.assertRaises(SystemExit) as error:
                infer.parse_args()
            self.assertNotEqual(error.exception.code, 0)
        result = subprocess.run(['bash', str(Path(__file__).with_name('run_all_gf3_henan_infer.sh')),
                                 '--threshold', '0.5'], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Unknown option: --threshold', result.stderr)

    def test_inference_config_copy(self):
        config = infer.Config(dict(
            model=dict(decode_head=dict(type='ChangeStarHead', seg_head_cfg={},
                                        out_channels=1, num_classes=2),
                       auxiliary_head=[dict(out_channels=1), dict(out_channels=2)]),
            visualizer=dict(type='CDLocalVisualizer', vis_backends=[dict(type='CDLocalVisBackend')]),
        ))
        modified = infer.prepare_inference_config(config)
        self.assertNotIn('threshold', config.model.decode_head)
        self.assertEqual(len(config.visualizer.vis_backends), 1)
        self.assertEqual(modified.visualizer.vis_backends, [])
        self.assertEqual(modified.model.decode_head.threshold, 0.3)
        self.assertEqual(modified.model.decode_head.seg_head_cfg.threshold, 0.3)
        self.assertEqual(modified.model.auxiliary_head[0].threshold, 0.3)
        self.assertNotIn('threshold', modified.model.auxiliary_head[1])

    def test_collate_default_and_custom(self):
        instance = object.__new__(infer.SceneInferencer)
        collate = instance._init_collate(infer.Config(dict(test_dataloader={})))
        self.assertIs(collate, infer.pseudo_collate)
        self.assertEqual(collate([{'value': 1}, {'value': 2}]), {'value': [1, 2]})
        cfg = infer.Config(dict(test_dataloader=dict(collate_fn='custom')))
        with patch.object(infer.OpenCDInferencer, '_init_collate', return_value='custom') as original:
            self.assertEqual(instance._init_collate(cfg), 'custom')
            original.assert_called_once_with(cfg)

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
                report, root, np.array([[0.8, 0.2], [0, 1.0]], dtype='float32'),
                np.array([[1, 1], [0, 2]], dtype='float32'), 0.5,
                data_root=root, config_path=root / 'config.py', checkpoint_path=root / 'model.pth',
                tile_output_dir=root / 'tile_png', total_tiles=10, used_tiles=2,
                decision_rule=infer.resolve_decision_rule(infer.Config(dict(model=dict(decode_head=dict(num_classes=2))))))
            with rasterio.open(root / 'change_binary.tif') as ds:
                np.testing.assert_array_equal(ds.read(1), [[1, 0], [255, 0]])
                self.assertEqual(ds.nodata, 255)
                self.assertEqual(ds.transform, transform)
            with rasterio.open(root / 'change_prob.tif') as ds:
                self.assertEqual(ds.nodata, -1)
                np.testing.assert_allclose(ds.read(1), [[0.8, 0.2], [-1, 0.5]])
            self.assertTrue((root / 'change_binary.png').is_file())
            self.assertTrue(json.loads((root / 'infer_report.json').read_text())['partial_run'])
            saved = json.loads((root / 'infer_report.json').read_text())
            self.assertEqual(saved['tile_threshold'], 0.5)
            self.assertEqual(saved['mosaic_threshold'], 0.5)
            self.assertEqual(saved['binarization_rule'], 'foreground_probability > threshold')


if __name__ == '__main__':
    unittest.main()
