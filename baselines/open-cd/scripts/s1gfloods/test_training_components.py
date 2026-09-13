"""训练优化组件的数值、采样及交付契约。"""
import csv
import json
import itertools
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from training_components import SharedSARRadiometric, ForegroundInfiniteSampler, FloodIoUMetric, sampling_plan
from select_checkpoint import select_checkpoint


class TrainingTests(unittest.TestCase):
    def test_shared_transform(self):
        im = np.arange(300, dtype=np.uint8).reshape(10, 10, 3)
        label = np.ones((10, 10), np.uint8)
        for seed in range(20):
            np.random.seed(seed)
            result = SharedSARRadiometric()(dict(img=[im.copy(), im.copy()], gt_seg_map=label))
            np.testing.assert_array_equal(*result['img'])
            self.assertIs(result['gt_seg_map'], label)
            self.assertEqual(result['img'][0].dtype, np.uint8)
        np.testing.assert_array_equal(im, np.arange(300, dtype=np.uint8).reshape(10, 10, 3))

    def test_sampling(self):
        ratios = {'a.png': 0., 'b.png': .1, 'c.png': .10001, 'd.png': 1.}
        plan = sampling_plan(ratios)
        self.assertEqual(plan['group_counts'], [1, 1, 2])
        class Dataset:
            data_root = '/tmp/data'
            data_prefix = dict(seg_map_path='/tmp/data/train/label')
            names = ['d.png', 'b.png', 'a.png', 'c.png']
            def __len__(self): return 4
            def get_data_info(self, i):
                return dict(seg_map_path='/tmp/data/train/label/' + self.names[i])
        ds = Dataset()
        sampler = ForegroundInfiniteSampler(ds, ratios, seed=42)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'sampling.json'
            path.write_text(json.dumps(plan))
            from_file = ForegroundInfiniteSampler(ds, sampling_file=str(path), seed=42)
            np.testing.assert_array_equal(from_file.weights, sampler.weights)
        expected = [plan['members'][n]['probability'] for n in ds.names]
        np.testing.assert_allclose(sampler.weights, expected)
        draws = list(itertools.islice(iter(sampler), 40000))
        self.assertEqual(draws[:100], list(itertools.islice(iter(sampler), 100)))
        np.testing.assert_allclose(np.bincount(draws, minlength=4)/40000, expected, atol=.01)
        with self.assertRaisesRegex(ValueError, 'membership'):
            ForegroundInfiniteSampler(ds, {'x.png': 0}, seed=42)
        self.assertEqual(sampling_plan({'a': 0})['group_probabilities'], [1., 0., 0.])
        for value in (-1, float('nan'), 1.1):
            with self.assertRaises(ValueError): sampling_plan({'a': value})

    def test_metric(self):
        metric = FloodIoUMetric(iou_metrics=['mIoU'])
        metric.dataset_meta = dict(classes=('background', 'change'))
        pred = torch.tensor([[0, 1, 1, 0, 1]])
        gt = torch.tensor([[0, 1, 0, 1, 255]])
        metric.process({}, [dict(pred_sem_seg=dict(data=pred), gt_sem_seg=dict(data=gt))])
        result = metric.compute_metrics(metric.results)
        self.assertAlmostEqual(result['FloodIoU'], 100/3)
        self.assertAlmostEqual(result['mIoU'], 33.33)
        row = metric.intersect_and_union(torch.zeros(3), torch.zeros(3), 2, 255)
        self.assertTrue(np.isnan(metric.compute_metrics([row])['FloodIoU']))

    def test_checkpoint_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)/'model'; work.mkdir()
            old = work/'best_mIoU_iter_2.pth'; old.touch()
            self.assertEqual(select_checkpoint(work), old)
            flood = work/'best_FloodIoU_iter_2.pth'; flood.touch()
            with self.assertRaises(ValueError): select_checkpoint(work)
            summary = work.parent/'summary.tsv'
            def write(primary, ckpt):
                with summary.open('w') as f:
                    writer = csv.DictWriter(f, ['model_tag', 'primary_metric', 'status', 'best_ckpt'], delimiter='\t')
                    writer.writeheader(); writer.writerow(dict(model_tag='model', primary_metric=primary,
                                                                status='success', best_ckpt=str(ckpt)))
            write('mIoU', old)
            self.assertEqual(select_checkpoint(work), old)
            write('FloodIoU', flood)
            self.assertEqual(select_checkpoint(work), flood)
            flood.unlink()
            with self.assertRaises(ValueError): select_checkpoint(work)


if __name__ == '__main__': unittest.main()
