#!/usr/bin/env python3
"""现场预检与训练编排；仅 full-train 启动完整训练。"""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
DINO = REPO / 'HA-CQI/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth'
TORCH_HOME = PROJECT / 'pretrained/torch'
MOBILE = TORCH_HOME / 'hub/checkpoints/mobilenet_v2-b0353104.pth'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def members(root: Path) -> dict:
    """现场文件名严格相等；不以历史 CSV 决定成员。"""
    result = {}
    for split in ('train', 'val'):
        sets = [{p.name for p in (root / split / part).glob('*.png')} for part in ('A', 'B', 'label')]
        if not sets[0] or not sets[0] == sets[1] == sets[2]:
            raise ValueError(f'{split}: empty or mismatched A/B/label')
        result[split] = []
        for name in sorted(sets[0]):
            files = []
            for part in ('A', 'B', 'label'):
                path = root / split / part / name
                st = path.stat()
                files.append(dict(path=str(path.relative_to(root)), size=st.st_size, mtime_ns=st.st_mtime_ns))
            result[split].append(dict(name=name, files=files))
    return result


def inspect_data(root: Path) -> tuple[dict, dict]:
    """原始 train A/B 像素联合统计，float64 累积后转换到 0–1 尺度。"""
    snapshot = members(root)
    sums, squares = np.zeros(3, np.float64), np.zeros(3, np.float64)
    count = 0
    for split, records in snapshot.items():
        for i, row in enumerate(records, 1):
            for part, info in zip(('A', 'B', 'label'), row['files']):
                path = root / info['path']
                with Image.open(path) as image:
                    if image.size != (256, 256) or image.mode != ('L' if part == 'label' else 'RGB'):
                        raise ValueError(f'Expected 256x256 RGB images / L labels: {path}')
                    array = np.asarray(image)
                    if part == 'label':
                        if not np.isin(array, [0, 255]).all():
                            raise ValueError(f'Expected 0/255 label: {path}')
                    elif split == 'train':
                        pixels = array.reshape(-1, 3).astype(np.float64)
                        sums += pixels.sum(0)
                        squares += np.square(pixels).sum(0)
                        count += len(pixels)
            if i % 1000 == 0:
                print(f'Checked {split}: {i}/{len(records)}', flush=True)
    if snapshot != members(root):
        raise ValueError('Dataset changed during preflight')
    mean = sums / count
    std = np.sqrt(np.maximum(squares / count - mean**2, 0))
    if not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError('Invalid training standard deviation')
    fingerprint = hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()
    stats = dict(data_root=str(root), split='train', scope='train_A_B_joint', scale='0_1',
                 accumulation_dtype='float64', num_pairs=len(snapshot['train']),
                 num_images=2*len(snapshot['train']), pixel_count=count, fingerprint=fingerprint,
                 recommended_config_fields=dict(mean=(mean/255).tolist(), std=(std/255).tolist()))
    return dict(data_root=str(root), members=snapshot, fingerprint=fingerprint,
                fingerprint_policy='relative_path_size_mtime_ns'), stats


def check_environment(dino: Path) -> dict:
    """只检查本地权重和 CUDA，不触发下载或模型初始化。"""
    sys.path.insert(0, str(PROJECT))
    report = dict(executable=sys.executable, versions={}, weights={})
    for name in ('torch', 'torchvision', 'timm', 'kornia', 'einops', 'rasterio', 'numpy'):
        module = importlib.import_module(name)
        report['versions'][name] = module.__version__
    import torch
    import model.create_ChangeDINO  # noqa: F401
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; CPU fallback disabled')
    x = torch.ones(1, device='cuda')
    assert (x+x).item() == 2
    report['gpu'] = torch.cuda.get_device_name(0)
    for path, prefix in ((dino, '08c60483'), (MOBILE, 'b0353104')):
        digest = sha256(path)
        if not digest.startswith(prefix):
            raise ValueError(f'Pretrained SHA256 mismatch: {path}')
        report['weights'][str(path)] = digest
    result = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True)
    report['pip_check'] = result.stdout + result.stderr
    if result.returncode:
        raise RuntimeError(report['pip_check'])
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['check-env', 'smoke-train', 'full-train'])
    p.add_argument('--dataset', default=os.getenv('DATASET_NAME', 'S1GFloods_CD_DINO_BG_75_25'))
    p.add_argument('--data-root', type=Path, default=Path(os.getenv('DATA_ROOT', str(REPO/'datasets'))))
    p.add_argument('--run-name', default=os.getenv('RUN_NAME'))
    p.add_argument('--checkpoint-root', type=Path, default=PROJECT/'checkpoints')
    p.add_argument('--dino-weight', type=Path, default=Path(os.getenv('DINO_WEIGHT', str(DINO))))
    p.add_argument('--batch-size', type=int, default=int(os.getenv('BATCH_SIZE', '12')))
    p.add_argument('--num-epochs', type=int, default=int(os.getenv('NUM_EPOCHS', '100')))
    p.add_argument('--lr', type=float, default=float(os.getenv('LR', '1e-4')))
    p.add_argument('--num-workers', type=int, default=int(os.getenv('NUM_WORKERS', '4')))
    p.add_argument('--seed', type=int, default=int(os.getenv('SEED', '42')))
    p.add_argument('--save-epoch-freq', type=int, default=int(os.getenv('SAVE_EPOCH_FREQ', '10')))
    args = p.parse_args(argv)
    if args.batch_size < 1 or args.num_epochs < 1 or args.lr <= 0 or args.num_workers < 0 or not 0 <= args.seed < 2**32:
        p.error('Invalid training hyperparameters')
    for name in (args.dataset, args.run_name):
        if name is not None and not re.fullmatch(r'[A-Za-z0-9_-]+', name):
            p.error('Dataset and run names must be plain directory names')
    for key in ('data_root', 'checkpoint_root', 'dino_weight'):
        setattr(args, key, (REPO / getattr(args, key)).resolve())
    return args


def run_path(args: argparse.Namespace) -> Path:
    """新目录保留隔离；显式同名拒绝，默认微秒时间戳避免覆盖。"""
    prefix = 'smoke-' if args.mode == 'smoke-train' else ''
    name = args.run_name or f'{prefix}S1GFloods-ChangeDINO-vits16-{datetime.now():%Y%m%d-%H%M%S-%f}'
    path = args.checkpoint_root / name
    if path.exists():
        raise FileExistsError(f'Run directory already exists: {path}')
    return path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ.update(TORCH_HOME=str(TORCH_HOME), NO_ALBUMENTATIONS_UPDATE='1')
    target = run_path(args) if args.mode != 'check-env' else None
    environment = check_environment(args.dino_weight)
    snapshot, stats = inspect_data(args.data_root / args.dataset)
    print(json.dumps(dict(environment=environment, stats=stats), indent=2), flush=True)
    if args.mode == 'check-env':
        return 0
    if args.mode == 'smoke-train' and len(snapshot['members']['train']) < 2*args.batch_size:
        raise ValueError('Smoke requires at least two full training batches')
    if snapshot['members'] != members(args.data_root / args.dataset):
        raise ValueError('Dataset changed after preflight')
    target.mkdir(parents=True, exist_ok=False)
    for filename, payload in [('normalization.json', stats), ('data_members.json', snapshot), ('environment.json', environment)]:
        (target/filename).write_text(json.dumps(payload, indent=2)+'\n')
    command = [sys.executable, str(PROJECT/'trainval.py'), '--name', target.name,
               '--checkpoint_dir', str(target.parent), '--dataset', args.dataset, '--dataroot', str(args.data_root),
               '--dataset_mode', 'sar', '--stats_file', str(target/'normalization.json'),
               '--dino_arch', 'dinov3_vits16', '--dino_weight', str(args.dino_weight),
               '--backbone', 'mobilenetv2', '--fpn_channels', '128', '--gpu_ids', '0',
               '--batch_size', str(args.batch_size), '--num_epochs', str(args.num_epochs),
               '--lr', str(args.lr), '--num_workers', str(args.num_workers), '--seed', str(args.seed),
               '--save_epoch_freq', str(args.save_epoch_freq)]
    if args.mode == 'smoke-train':
        command.append('--smoke_train')
    (target/'launch.json').write_text(json.dumps(dict(mode=args.mode, command=command), indent=2)+'\n')
    print(f'Run directory: {target}', flush=True)
    with (target/'train.log').open('w') as log:
        process = subprocess.Popen(command, cwd=PROJECT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, errors='replace')
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line); log.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
            raise
    best = list(target.glob('*_best.pth'))
    if code == 0 and len(best) != 1:
        code = 1
    (target/'run_status.json').write_text(json.dumps(dict(exit_code=code, best=[str(p) for p in best]), indent=2)+'\n')
    return code


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'[ERROR] {exc}', file=sys.stderr)
        raise SystemExit(1)
