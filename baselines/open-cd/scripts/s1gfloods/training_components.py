"""S1GFloods 专用增强、温和分层采样及前景指标。"""
from __future__ import annotations

from pathlib import Path
from collections.abc import Iterator
from typing import Any
import itertools
import json
import numpy as np
from mmcv.transforms import BaseTransform
from mmengine.dist import get_dist_info, sync_random_seed
from mmseg.evaluation import IoUMetric
from torch.utils.data import Sampler
from opencd.registry import TRANSFORMS, DATA_SAMPLERS, METRICS


def sampling_plan(ratios: dict[str, float]) -> dict:
    """按原始标签分组，仅对非空组混合原分布与均匀分布。"""
    names = sorted(ratios)
    values = np.array([ratios[n] for n in names], dtype=np.float64)
    if not len(values) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('Invalid foreground ratios')
    groups = np.where(values == 0, 0, np.where(values <= .1, 1, 2))
    counts = np.bincount(groups, minlength=3)
    probs = np.where(counts > 0, .5 * counts / len(names) + .5 / np.count_nonzero(counts), 0)
    return dict(group_counts=counts.tolist(), group_probabilities=probs.tolist(),
                members={n: dict(foreground_ratio=float(v), group=int(g),
                                probability=float(probs[g] / counts[g]))
                         for n, v, g in zip(names, values, groups)})


@TRANSFORMS.register_module()
class SharedSARRadiometric(BaseTransform):
    """双时相共享亮度及对比度随机变量，标签不参与处理。"""
    def transform(self, results: dict) -> dict:
        delta = np.random.uniform(-10, 10) if np.random.rand() < .5 else 0.
        alpha = np.random.uniform(.8, 1.2) if np.random.rand() < .5 else 1.
        results['img'] = [np.clip(np.clip(im.astype(np.float32) + delta, 0, 255)
                                  * alpha, 0, 255).astype(im.dtype)
                          for im in results['img']]
        return results


@DATA_SAMPLERS.register_module()
class ForegroundInfiniteSampler(Sampler):
    """按文件名匹配概率；组内均匀有放回抽样，支持无限迭代。"""
    def __init__(self, dataset: Any, ratios: dict[str, float] | None = None,
                 seed: int | None = None, sampling_file: str | None = None) -> None:
        if (ratios is None) == (sampling_file is None):
            raise ValueError('Provide exactly one of ratios or sampling_file')
        if sampling_file is not None:
            with open(sampling_file) as stream:
                records = json.load(stream)['members']
            ratios = {name: row['foreground_ratio'] for name, row in records.items()}
        self.dataset = dataset
        self.seed = sync_random_seed() if seed is None else seed
        self.rank, self.world_size = get_dist_info()
        plan = sampling_plan(ratios)
        root = Path(dataset.data_root) / dataset.data_prefix['seg_map_path']
        names = [Path(dataset.get_data_info(i)['seg_map_path']).relative_to(root).as_posix()
                 for i in range(len(dataset))]
        if len(set(names)) != len(names) or set(names) != set(ratios):
            raise ValueError('Sampler filename membership mismatch')
        self.weights = np.array([plan['members'][n]['probability'] for n in names])

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed)
        def stream():
            while True:
                yield from rng.choice(len(self.dataset), size=1024, p=self.weights).tolist()
        return itertools.islice(stream(), self.rank, None, self.world_size)

    def __len__(self) -> int:
        return len(self.dataset)

    def set_epoch(self, epoch: int) -> None:
        pass  # IterBasedTrainLoop 使用同一个连续随机流。


@METRICS.register_module()
class FloodIoUMetric(IoUMetric):
    """保留原指标并输出未舍入的前景 IoU；无并集时不伪造分数。"""
    def compute_metrics(self, results: list) -> dict:
        metrics = super().compute_metrics(results)
        if self.format_only:
            return metrics
        index = list(self.dataset_meta['classes']).index('change')
        intersection = sum(float(row[0][index]) for row in results)
        union = sum(float(row[1][index]) for row in results)
        metrics['FloodIoU'] = 100. * intersection / union if union > 0 else float('nan')
        return metrics
