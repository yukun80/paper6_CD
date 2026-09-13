#!/usr/bin/env python3
"""按现场数据重算统计，预检后串行启动 Open-CD；不安装依赖或下载权重。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime
from typing import Callable

import numpy as np
from PIL import Image

OPENCD = Path(__file__).resolve().parents[2]
MODELS = {
    'fc_siam_diff': 'fcsn/fc_siam_diff_256x256_40k_s1gfloods.py',
    'ifn': 'ifn/ifn_256x256_40k_s1gfloods.py',
    'bit': 'bit/bit_r18_256x256_40k_s1gfloods.py',
    'changestar': 'changestar/changestar_farseg_1x96_256x256_40k_s1gfloods.py',
    'lightcdnet': 'lightcdnet/lightcdnet_s_256x256_40k_s1gfloods.py',
    'changeformer': 'changeformer/changeformer_mit-b0_256x256_40k_s1gfloods.py',
    'cgnet': 'cgnet/cgnet_256x256_40k_s1gfloods.py',
    'snunet': 'snunet/snunet_c16_256x256_40k_s1gfloods.py',
    'hanet': 'hanet/hanet_256x256_40k_s1gfloods.py',
    'ttp': 'ttp/ttp_vit-sam-b_256x256_40k_s1gfloods.py',
}
EXTRA = list(MODELS)[5:]
VGG = 'vgg16-397923af.pth'
RESNET = 'resnet18_v1c-b5776b93.pth'
SAM = 'vit-base-p16_sam-pre_3rdparty_sa1b-1024px_20230411-2320f9cc.pth'


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """兼容旧模式入口，新增可审阅命令和显式模型子集。"""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['check-env', 'smoke-train', 'full-train'])
    p.add_argument('--group', choices=['all', 'extra'], default='all')
    p.add_argument('--models', nargs='+', choices=list(MODELS))
    p.add_argument('--data-root', type=Path,
                   default=OPENCD.parent.parent / 'datasets/S1GFloods_CD_DINO_BG_75_25')
    p.add_argument('--batch-root', type=Path, default=OPENCD / 'work_dirs')
    p.add_argument('--torch-home', type=Path, default=OPENCD / 'pretrained/torch')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--gpus', type=int, choices=[1], default=1)
    p.add_argument('--save-best', choices=['FloodIoU', 'mIoU'], default='FloodIoU')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args(argv)
    if not 0 <= args.seed < 2**32:
        p.error('--seed must be within [0, 2**32)')
    for key in ('data_root', 'batch_root', 'torch_home'):
        setattr(args, key, getattr(args, key).resolve())
    allowed = EXTRA if args.group == 'extra' else list(MODELS)
    if args.models and any(m not in allowed for m in args.models):
        p.error('--group extra only accepts: ' + ' '.join(EXTRA))
    args.models = [m for m in allowed if not args.models or m in args.models]
    return args


def collect_members(root: Path) -> dict[str, list[dict]]:
    """用目录中的精确 PNG 配对构建快照，不依赖旧 manifest。"""
    result = {}
    for split in ('train', 'val'):
        folders = [root / split / name for name in ('A', 'B', 'label')]
        if any(not folder.is_dir() for folder in folders):
            raise ValueError(f'Missing A/B/label directory in {root / split}')
        sets = [{p.relative_to(folder).as_posix() for p in folder.rglob('*.png')}
                for folder in folders]
        if not sets[0] or not sets[0] == sets[1] == sets[2]:
            raise ValueError(f'{split}: A/B/label filename mismatch or empty split: '
                             f'{[len(s) for s in sets]}')
        records = []
        for name in sorted(sets[0]):
            files = []
            for folder in folders:
                path = folder / name
                st = path.stat()
                files.append(dict(path=path.relative_to(root).as_posix(),
                                  size=st.st_size, mtime_ns=st.st_mtime_ns))
            records.append(dict(name=name, files=files))
        result[split] = records
    return result


def inspect_data(root: Path) -> tuple[dict, dict]:
    """全量检查格式；仅联合训练集双时相像素计算总体标准差。"""
    members = collect_members(root)
    sums = np.zeros(3, dtype=np.float64)
    squares = np.zeros(3, dtype=np.float64)
    count = 0
    ratios = {}
    for split, records in members.items():
        for pair_index, row in enumerate(records, 1):
            for index, info in enumerate(row['files']):
                path = root / info['path']
                with Image.open(path) as im:
                    expected = 'L' if index == 2 else 'RGB'
                    if im.mode != expected or im.size != (256, 256):
                        raise ValueError(f'Expected {expected} 256x256: {path}')
                    array = np.asarray(im)
                    if index == 2:
                        if not np.isin(array, [0, 255]).all():
                            raise ValueError(f'Label is not 0/255: {path}')
                        if split == 'train':
                            ratios[row['name']] = float(np.count_nonzero(array) / array.size)
                    elif split == 'train':
                        pixels = array.reshape(-1, 3).astype(np.float64)
                        sums += pixels.sum(axis=0)
                        squares += np.square(pixels).sum(axis=0)
                        count += len(pixels)
            if pair_index % 1000 == 0:
                print(f'Checked {split}: {pair_index}/{len(records)} pairs', flush=True)
    if members != collect_members(root):
        raise ValueError('Dataset changed while computing statistics')
    mean = sums / count
    std = np.sqrt(np.maximum(squares / count - np.square(mean), 0))
    if not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError('Invalid or zero training standard deviation')
    digest = hashlib.sha256(json.dumps(members, sort_keys=True).encode()).hexdigest()
    snapshot = dict(data_root=str(root), counts={k: len(v) for k, v in members.items()},
                    fingerprint_policy='relative_path_size_mtime_ns',
                    fingerprint=digest, members=members)
    stats = dict(scope='train_A_B_joint', scale='0_255', dtype='float64',
                 pixel_count_per_channel=count, mean=mean.tolist(), std=std.tolist(),
                 six_channel_mean=mean.tolist() * 2, six_channel_std=std.tolist() * 2,
                 dataset_fingerprint=digest)
    snapshot['foreground_ratios'] = ratios
    return snapshot, stats


def assets_for(models: list[str], torch_home: Path) -> dict[str, Path]:
    """IFN 的硬编码 torchvision 初始化也必须命中相同 Torch 缓存。"""
    cache = torch_home / 'hub/checkpoints'
    assets = {}
    if 'ifn' in models:
        assets['vgg16'] = cache / VGG
    if {'bit', 'changestar'} & set(models):
        assets['resnet18'] = cache / RESNET
    if 'ttp' in models:
        assets['sam'] = OPENCD / 'pretrained' / SAM
    return assets


def check_assets(assets: dict[str, Path]) -> list[str]:
    """验证文件及官方文件名中的摘要前缀，不触发下载。"""
    errors = []
    for name, path in assets.items():
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f'Missing pretrained asset {name}: {path}')
            continue
        digest = file_sha256(path)
        prefix = path.stem.rsplit('-', 1)[-1]
        if not digest.startswith(prefix):
            errors.append(f'Pretrained SHA256 prefix mismatch: {path}')
    return errors


def file_sha256(path: Path) -> str:
    """分块读取大权重，兼容 Python 3.10。"""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check_environment(models: list[str]) -> tuple[dict, list[str]]:
    """收集全部依赖错误，明确检查被可选导入隐藏的 TTP 注册。"""
    from mmengine.utils import digit_version
    errors = []
    report = dict(python=sys.version, executable=sys.executable, versions={})
    sys.path.insert(0, str(OPENCD))
    required = ['torch', 'torchvision', 'mmcv', 'mmengine', 'mmseg', 'mmdet', 'opencd']
    if 'ttp' in models:
        required += ['mmpretrain', 'transformers', 'peft', 'accelerate', 'einops']
    for name in required:
        try:
            report['versions'][name] = getattr(importlib.import_module(name), '__version__', 'unknown')
        except Exception as exc:
            errors.append(f'{name}: {type(exc).__name__}: {exc}')
    bounds = {'mmcv': ('2.0.0rc4', '2.2.0'), 'mmengine': ('0.8.3', '1.0.0'),
              'mmseg': ('1.2.2', '1.3.0'), 'mmdet': ('3.0.0rc6', '4.0.0')}
    for name, (low, high) in bounds.items():
        version = report['versions'].get(name)
        if version and not digit_version(low) <= digit_version(version) < digit_version(high):
            errors.append(f'{name}=={version}; required >= {low}, < {high}')
    try:
        import torch
        from mmcv.ops import nms
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; CPU training fallback is disabled')
        nms(torch.tensor([[0., 0., 1., 1.]], device='cuda'),
            torch.tensor([1.], device='cuda'), 0.5)
        report['gpu'] = dict(name=torch.cuda.get_device_name(0),
                             memory_free_total=list(torch.cuda.mem_get_info(0)))
    except Exception as exc:
        errors.append(f'CUDA/ops: {type(exc).__name__}: {exc}')
    try:
        import mmseg.models  # noqa: F401
        import mmdet.models  # noqa: F401
        import opencd.models  # noqa: F401
        import opencd.datasets  # noqa: F401
        import scripts.s1gfloods.training_components  # noqa: F401
        from mmengine.config import Config
        from mmengine.registry import init_default_scope
        from opencd.registry import MODELS as registry
        init_default_scope('opencd')
        for model in models:
            config = Config.fromfile(OPENCD / 'configs' / MODELS[model])
            for component in (config.model, config.model.backbone,
                              config.model.decode_head):
                if registry.get(component['type']) is None:
                    raise RuntimeError(f'{model}: unregistered {component["type"]}')
        if 'ttp' in models:
            import mmpretrain.models  # noqa: F401
            from transformers.modeling_utils import (apply_chunking_to_forward,
                find_pruneable_heads_and_indices, prune_linear_layer)  # noqa: F401
            for name in ('VisionTransformerTurner', 'ViTSAM_Custom'):
                if registry.get(name) is None:
                    raise RuntimeError(f'Missing registered TTP model: {name}')
    except Exception as exc:
        errors.append(f'Model registration: {type(exc).__name__}: {exc}')
    pip = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True)
    report['pip_check'] = pip.stdout + pip.stderr
    if pip.returncode:
        errors.append('pip check failed: ' + report['pip_check'].strip())
    return report, errors


def make_config(model: str, args: argparse.Namespace, stats: dict, ratios: dict, sampling_file=None):
    """生成 SAR 优化配置；短验证不降低训练 batch。"""
    from mmengine.config import Config
    cfg = Config.fromfile(OPENCD / 'configs' / MODELS[model])
    cfg.data_root = str(args.data_root)
    cfg.load_from = None
    cfg.resume = False
    cfg.randomness = dict(seed=args.seed, deterministic=False, diff_rank_seed=False)
    for split in ('train', 'val', 'test'):
        ds = cfg[f'{split}_dataloader']['dataset']
        ds['data_root'] = str(args.data_root)
        ds['ann_file'] = ''
        ds['format_seg_map'] = 'to_binary'
    for pre in (cfg.model.data_preprocessor, cfg.get('data_preprocessor', {})):
        pre['mean'] = stats['six_channel_mean']
        pre['std'] = stats['six_channel_std']
    imports = list(cfg.get('custom_imports', {}).get('imports', []))
    imports.append('scripts.s1gfloods.training_components')
    cfg.custom_imports = dict(imports=list(dict.fromkeys(imports)), allow_failed_imports=False)
    pipeline = []
    for transform in cfg.train_dataloader.dataset.pipeline:
        if transform['type'] == 'MultiImgRandomCrop':
            continue
        if transform['type'] == 'MultiImgPhotoMetricDistortion':
            transform = dict(type='SharedSARRadiometric')
        pipeline.append(transform)
    cfg.train_pipeline = pipeline
    cfg.train_dataloader.dataset.pipeline = pipeline
    cfg.train_dataloader.sampler = dict(type='ForegroundInfiniteSampler', ratios=ratios,
                                        seed=args.seed)
    if sampling_file is not None:
        cfg.train_dataloader.sampler.pop('ratios')
        cfg.train_dataloader.sampler.sampling_file = str(sampling_file)
    learning_rates = dict(fc_siam_diff=5e-4, ifn=5e-4, bit=5e-4, snunet=5e-4,
                          hanet=5e-4, lightcdnet=1e-3)
    if model in learning_rates:
        cfg.optim_wrapper.optimizer.lr = learning_rates[model]
        if 'optimizer' in cfg:
            cfg.optimizer.lr = learning_rates[model]
    for split in ('val', 'test'):
        cfg[f'{split}_evaluator']['type'] = 'FloodIoUMetric'
    cfg.default_hooks.checkpoint.save_best = ['FloodIoU', 'mIoU']
    cfg.primary_metric = args.save_best
    cfg.default_hooks.checkpoint.rule = 'greater'
    assets = assets_for([model], args.torch_home)
    if model == 'bit':
        cfg.model.pretrained = str(assets['resnet18'])
    elif model == 'changestar':
        cfg.model.backbone.init_cfg.checkpoint = str(assets['resnet18'])
    elif model == 'ttp':
        cfg.model.backbone.encoder_cfg.init_cfg.checkpoint = str(assets['sam'])
    if args.mode == 'smoke-train':
        cfg.train_cfg.max_iters = 2
        cfg.train_cfg.val_interval = 2
        cfg.default_hooks.checkpoint.interval = 2
        for split in ('val', 'test'):
            cfg[f'{split}_dataloader']['dataset']['indices'] = 2
    return cfg


def batch_path(root: Path, mode: str) -> Path:
    """短验证使用独立前缀，避免被既有推理脚本当作最新正式批次。"""
    prefix = 's1gfloods-smoke' if mode == 'smoke-train' else 's1gfloods-batch'
    base = root / f'{prefix}-{datetime.now():%Y%m%d-%H%M%S}'
    path, index = base, 1
    while path.exists():
        path = Path(f'{base}-{index}')
        index += 1
    return path


def child_env(torch_home: Path, batch: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(TORCH_HOME=str(torch_home), NO_ALBUMENTATIONS_UPDATE='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               MPLCONFIGDIR=str(batch / 'mplconfig'),
               PYTHONPATH=str(OPENCD) + os.pathsep + env.get('PYTHONPATH', ''))
    return env


def run_process(command: list[str], log_path: Path, env: dict[str, str]) -> int:
    """同步写入终端与日志；中断时终止整个训练进程组。"""
    with log_path.open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=OPENCD, env=env, start_new_session=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, errors='replace')
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            return process.wait()
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise
        finally:
            process.stdout.close()


def execute_jobs(jobs: list[dict], batch: Path, env: dict[str, str],
                 runner: Callable = run_process) -> int:
    """汇总真实进程状态；失败后继续，但中断绝不推进下一模型。"""
    fields = ['model_tag', 'status', 'exit_code', 'config', 'work_dir', 'best_ckpt',
              'primary_metric', 'best_flood_ckpt', 'best_miou_ckpt', 'log_file', 'started_at', 'ended_at', 'duration_sec']
    failures = 0
    with (batch / 'summary.tsv').open('w', newline='') as stream, \
         (batch / 'succeeded_models.txt').open('w') as success, \
         (batch / 'failed_models.txt').open('w') as failed:
        writer = csv.DictWriter(stream, fields, delimiter='\t')
        writer.writeheader()
        for job in jobs:
            started = time.time()
            status = 'failed'
            try:
                code = runner(job['command'], Path(job['log_file']), env)
            except KeyboardInterrupt:
                code, status = 130, 'interrupted'
            except Exception as exc:
                code = 1
                with Path(job['log_file']).open('a') as log:
                    log.write(f'Launcher error: {type(exc).__name__}: {exc}\n')
            if code in (130, 143, -signal.SIGINT, -signal.SIGTERM):
                status = 'interrupted'
            primary = job.get('primary_metric', 'FloodIoU')
            paths = {metric: sorted(Path(job['work_dir']).glob(f'best_{metric}_*.pth'))
                     for metric in ('FloodIoU', 'mIoU')}
            best = paths[primary]
            if len(best) > 1:
                code = 1  # 不允许在歧义权重间任意选择。

            if code == 0 and best:
                status = 'success'
                success.write(job['model_tag'] + '\n')
                success.flush()
            else:
                if code == 0:
                    code = 1  # 进程退出成功但没有 best checkpoint 不能报成功。
                failed.write(job['model_tag'] + '\n')
                failed.flush()
                failures += 1
            row = {k: job[k] for k in ('model_tag', 'config', 'work_dir', 'log_file')}
            row.update(primary_metric=primary,
                       best_flood_ckpt=str(paths['FloodIoU'][0]) if len(paths['FloodIoU']) == 1 else '',
                       best_miou_ckpt=str(paths['mIoU'][0]) if len(paths['mIoU']) == 1 else '',
                       status=status, exit_code=code, best_ckpt=str(best[0]) if best else '',
                       started_at=datetime.fromtimestamp(started).isoformat(),
                       ended_at=datetime.now().isoformat(), duration_sec=round(time.time()-started, 3))
            writer.writerow(row)
            stream.flush()
            if status == 'interrupted':
                return 130
    return int(failures > 0)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sys.path.insert(0, str(OPENCD))
    os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
    os.environ['TORCH_HOME'] = str(args.torch_home)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    print('Inspecting live train/val files and recomputing train-only statistics...', flush=True)
    snapshot, stats = inspect_data(args.data_root)
    print(json.dumps(dict(counts=snapshot['counts'], stats=stats), indent=2))
    assets = assets_for(args.models, args.torch_home)
    errors = check_assets(assets)
    if args.dry_run:
        environment = dict(executable=sys.executable, runtime_check='not_run_in_dry_run')
    else:
        environment, env_errors = check_environment(args.models)
        errors.extend(env_errors)
    batch = batch_path(args.batch_root, args.mode)
    configs = {m: make_config(m, args, stats, snapshot['foreground_ratios'], batch / 'sampling.json')
               for m in args.models}
    if args.mode == 'smoke-train':
        for cfg in configs.values():
            for split in ('val', 'test'):
                cfg[f'{split}_dataloader']['dataset']['indices'] = min(2, snapshot['counts']['val'])
    for error in errors:
        print('[BLOCKED] ' + error, file=sys.stderr)
    if args.mode == 'check-env' and not args.dry_run:
        print(json.dumps(environment, indent=2))
        return int(bool(errors))
    if errors and not args.dry_run:
        return 1
    jobs = []
    for name, cfg in configs.items():
        tag = Path(MODELS[name]).stem
        work = batch / tag
        config_path = work / f'{tag}.py'
        cfg.work_dir = str(work)
        command = [sys.executable, str(OPENCD / 'tools/train.py'), str(config_path),
                   '--work-dir', str(work)]
        jobs.append(dict(model_tag=tag, primary_metric=args.save_best, config=str(config_path), work_dir=str(work),
                         log_file=str(batch / 'logs' / f'{tag}.log'), command=command))
        if args.dry_run:
            print(f'\n# EFFECTIVE CONFIG {name}\n{cfg.pretty_text}')
            environment_values = child_env(args.torch_home, batch)
            env_preview = {key: environment_values[key] for key in (
                'TORCH_HOME', 'PYTHONPATH', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE',
                'NO_ALBUMENTATIONS_UPDATE', 'MPLCONFIGDIR')}
            print('# COMMAND (configuration is a preview; no files created)\n' +
                  ' '.join(f'{k}={shlex.quote(v)}' for k, v in env_preview.items()) +
                  ' ' + shlex.join(command))
    if args.dry_run:
        print('Dry-run completed; no runtime readiness or training success claimed.')
        return 0
    if snapshot['members'] != collect_members(args.data_root):
        raise ValueError('Dataset changed after preflight')
    batch.mkdir(parents=True, exist_ok=False)
    (batch / 'logs').mkdir()
    for name, job in zip(configs, jobs):
        Path(job['work_dir']).mkdir()
        configs[name].dump(job['config'])
    environment['assets'] = {k: dict(path=str(v), sha256=file_sha256(v)) for k, v in assets.items()}
    write_json(batch / 'environment.json', environment)
    write_json(batch / 'data_members.json', snapshot)
    write_json(batch / 'normalization.json', stats)
    from scripts.s1gfloods.training_components import sampling_plan
    write_json(batch / 'sampling.json', sampling_plan(snapshot['foreground_ratios']))
    write_json(batch / 'batch_plan.json', dict(mode=args.mode, seed=args.seed, jobs=jobs))
    print(f'Batch directory: {batch}', flush=True)
    return execute_jobs(jobs, batch, child_env(args.torch_home, batch))


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f'[ERROR] {type(exc).__name__}: {exc}', file=sys.stderr)
        raise SystemExit(1)
