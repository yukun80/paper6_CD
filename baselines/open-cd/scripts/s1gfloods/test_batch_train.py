"""验证统计与编排契约；使用临时数据及子进程，不构建或训练模型。"""
import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import batch_train as b


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def dataset(self):
        # A/B 两时相分布不同；val 极值不得影响 train 统计。
        for split in ('train', 'val'):
            for component, value in [('A', 10), ('B', 30), ('label', 255)]:
                folder = self.root / split / component
                folder.mkdir(parents=True)
                if component == 'label':
                    array = np.full((256, 256), value, np.uint8)
                else:
                    channels = [value, value + 5, value + 10] if split == 'train' else [250]*3
                    array = np.broadcast_to(np.array(channels, np.uint8), (256, 256, 3))
                Image.fromarray(array).save(folder / 'one.png')
        return self.root

    def test_joint_stats_train_only(self):
        root = self.dataset()
        snapshot, stats = b.inspect_data(root)
        np.testing.assert_allclose(stats['mean'], [20, 25, 30])
        np.testing.assert_allclose(stats['std'], [10, 10, 10])
        self.assertEqual(stats['six_channel_mean'], [20, 25, 30]*2)
        self.assertEqual(stats['pixel_count_per_channel'], 2*256*256)
        self.assertEqual(snapshot['counts'], {'train': 1, 'val': 1})
        self.assertEqual(snapshot['foreground_ratios'], {'one.png': 1.})

    def test_missing_pair_rejected(self):
        root = self.dataset()
        (root / 'train/B/one.png').unlink()
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            b.inspect_data(root)

    def test_invalid_label_rejected(self):
        root = self.dataset()
        Image.fromarray(np.ones((256, 256), np.uint8)).save(root / 'val/label/one.png')
        with self.assertRaisesRegex(ValueError, '0/255'):
            b.inspect_data(root)

    def test_model_configs_and_defaults(self):
        args = b.parse_args(['full-train'])
        self.assertEqual(len(args.models), 10)
        self.assertNotIn('changer', args.models)
        self.assertNotIn('stanet', args.models)
        with self.assertRaises(SystemExit):
            b.parse_args(['full-train', '--models', 'stanet'])
        self.assertEqual(args.save_best, 'FloodIoU')
        self.assertEqual(args.seed, 42)
        self.assertEqual(b.parse_args(['--group', 'extra', 'full-train']).models, b.EXTRA)
        stats = {'six_channel_mean': [20, 25, 30]*2, 'six_channel_std': [10]*6}
        from mmengine.config import Config
        for model in args.models:
            old = Config.fromfile(b.OPENCD / 'configs' / b.MODELS[model])
            cfg = b.make_config(model, args, stats, {"one.png": 1.})
            self.assertEqual(cfg.optim_wrapper.optimizer.weight_decay, old.optim_wrapper.optimizer.weight_decay)
            expected_lr = dict(fc_siam_diff=5e-4, ifn=5e-4, bit=5e-4, snunet=5e-4,
                               hanet=5e-4, lightcdnet=1e-3, changestar=1e-3,
                               changeformer=6e-5, cgnet=5e-4, ttp=4e-4)[model]
            self.assertEqual(cfg.optim_wrapper.optimizer.lr, expected_lr)
            self.assertEqual(cfg.optim_wrapper.get('paramwise_cfg'), old.optim_wrapper.get('paramwise_cfg'))
            self.assertEqual(cfg.model.decode_head, old.model.decode_head)
            self.assertEqual(cfg.val_dataloader.sampler, old.val_dataloader.sampler)
            self.assertEqual(cfg.val_dataloader.dataset.pipeline, old.val_dataloader.dataset.pipeline)
            self.assertEqual(cfg.train_dataloader.sampler.type, 'ForegroundInfiniteSampler')
            self.assertEqual(cfg.param_scheduler, old.param_scheduler)
            self.assertNotIn('MultiImgRandomCrop', [t['type'] for t in cfg.train_pipeline])
            self.assertEqual(cfg.train_cfg, old.train_cfg)
            self.assertEqual(cfg.train_dataloader.batch_size, 2 if model == 'ttp' else 8)
            file_cfg = b.make_config(model, args, stats, {'one.png': 1.}, self.root/'sampling.json')
            self.assertNotIn('ratios', file_cfg.train_dataloader.sampler)
            self.assertEqual(file_cfg.train_dataloader.sampler.sampling_file, str(self.root/'sampling.json'))
            self.assertEqual(cfg.randomness.seed, 42)
            self.assertFalse(cfg.resume)
            self.assertIsNone(cfg.load_from)
            self.assertEqual(cfg.model.data_preprocessor.mean, stats['six_channel_mean'])
            self.assertEqual(cfg.default_hooks.checkpoint.save_best, ['FloodIoU', 'mIoU'])
            for split in ('train', 'val', 'test'):
                ds = cfg[f'{split}_dataloader']['dataset']
                self.assertEqual(ds.data_root, str(args.data_root))
                self.assertEqual(ds.format_seg_map, 'to_binary')
            args.mode = 'smoke-train'
            smoke = b.make_config(model, args, stats, {"one.png": 1.})
            self.assertEqual(smoke.train_dataloader.batch_size, cfg.train_dataloader.batch_size)
            self.assertEqual(smoke.train_cfg.max_iters, 2)
            self.assertEqual(smoke.val_dataloader.dataset.indices, 2)
            args.mode = 'full-train'

    def jobs(self, codes):
        (self.root / 'logs').mkdir()
        jobs = []
        for index, code in enumerate(codes):
            tag = f'model{index}'
            work = self.root / tag
            work.mkdir()
            # 真正执行小型子进程，成功时模拟训练交付的 best 文件。
            source = f'from pathlib import Path; import sys; '
            if code == 0:
                source += f'Path({str(work / "best_FloodIoU_iter_40000.pth")!r}).touch(); '
            source += f'print("fixture {index}"); sys.exit({code})'
            jobs.append(dict(model_tag=tag, config=str(work / 'config.py'),
                             work_dir=str(work), log_file=str(self.root / 'logs' / f'{tag}.log'),
                             command=[sys.executable, '-c', source]))
        return jobs

    def test_failure_continues_and_summary(self):
        jobs = self.jobs([1, 0])
        code = b.execute_jobs(jobs, self.root, b.child_env(self.root, self.root))
        self.assertEqual(code, 1)
        with (self.root / 'summary.tsv').open() as f:
            rows = list(csv.DictReader(f, delimiter='\t'))
        self.assertEqual([r['status'] for r in rows], ['failed', 'success'])
        self.assertEqual((self.root / 'succeeded_models.txt').read_text(), 'model1\n')
        self.assertIn('fixture 1', Path(jobs[1]['log_file']).read_text())

    def test_all_success(self):
        jobs = self.jobs([0, 0])
        self.assertEqual(b.execute_jobs(jobs, self.root, b.child_env(self.root, self.root)), 0)

    def test_interruption_stops_queue(self):
        jobs = self.jobs([0, 0])
        with patch.object(b, 'run_process') as runner:
            runner.side_effect = KeyboardInterrupt
            self.assertEqual(b.execute_jobs(jobs, self.root, {}, runner=runner), 130)
            self.assertEqual(runner.call_count, 1)

    def test_missing_best_is_failure(self):
        jobs = self.jobs([0])
        self.assertEqual(b.execute_jobs(jobs, self.root, {}, runner=lambda *args: 0), 1)

    def test_dry_run_creates_no_outputs(self):
        root = self.dataset()
        output = root / 'outputs'
        result = subprocess.run([sys.executable, str(Path(b.__file__)), 'full-train',
                                 '--dry-run', '--models', 'fc_siam_diff', '--data-root', str(root),
                                 '--batch-root', str(output)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(output.exists())
        self.assertIn('seed=42', result.stdout)
        self.assertIn('COMMAND', result.stdout)

    def test_missing_asset_rejected(self):
        errors = b.check_assets(b.assets_for(['ifn', 'bit'], self.root))
        self.assertEqual(len(errors), 2)

    def test_global_preflight_blocks_before_batch_creation(self):
        root = self.dataset()
        output = root / 'outputs'
        with patch.object(b, 'check_environment', return_value=({}, ['missing dependency'])), \
             patch.object(b, 'execute_jobs') as execute:
            code = b.main(['full-train', '--models', 'fc_siam_diff',
                           '--data-root', str(root), '--batch-root', str(output)])
        self.assertEqual(code, 1)
        execute.assert_not_called()
        self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
