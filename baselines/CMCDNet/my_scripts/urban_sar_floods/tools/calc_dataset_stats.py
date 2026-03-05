#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import rasterio


def read_split_ids(list_path: Path):
    ids = []
    for line in list_path.read_text(encoding='utf-8').splitlines():
        s = line.strip()
        if s:
            ids.append(s)
    return ids


def compute(root: Path, split: str, split_file: Path):
    sample_ids = read_split_ids(split_file)
    n_channels = 8

    count = 0
    sum_c = np.zeros(n_channels, dtype=np.float64)
    sq_sum_c = np.zeros(n_channels, dtype=np.float64)
    pixel_count = 0

    label_hist = np.zeros(3, dtype=np.int64)

    for sid in sample_ids:
        a = np.load(root / split / 'A_npy' / f'{sid}.npy').astype(np.float32, copy=False)
        b = np.load(root / split / 'B_npy' / f'{sid}.npy').astype(np.float32, copy=False)
        x = np.concatenate([a, b], axis=0)  # (8,H,W)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        x2 = x.reshape(n_channels, -1)
        sum_c += x2.sum(axis=1)
        sq_sum_c += np.square(x2).sum(axis=1)
        pixel_count += x2.shape[1]

        with rasterio.open(root / split / 'label' / f'{sid}.tif') as ds:
            y = ds.read(1)
        y = y.astype(np.uint8, copy=False)
        label_hist += np.bincount(y.ravel(), minlength=3)[:3]

        count += 1

    mean = sum_c / pixel_count
    var = sq_sum_c / pixel_count - np.square(mean)
    std = np.sqrt(np.maximum(var, 1e-12))

    freqs = label_hist / label_hist.sum()
    # ENet-style class balancing.
    class_weight = 1.0 / np.log(1.02 + freqs)

    return {
        'split': split,
        'samples': count,
        'pixels_per_channel': int(pixel_count),
        'channel_mean': mean.tolist(),
        'channel_std': std.tolist(),
        'label_hist': label_hist.tolist(),
        'label_freq': freqs.tolist(),
        'class_weight_enet': class_weight.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--split-file', type=Path, default=None)
    parser.add_argument('--save-json', type=Path, default=None)
    args = parser.parse_args()

    split_file = args.split_file or (args.data_root / 'meta' / f'{args.split}_list.txt')
    out = compute(args.data_root, args.split, split_file)

    print('samples:', out['samples'])
    print('label_hist:', out['label_hist'])
    print('label_freq:', out['label_freq'])
    print('class_weight_enet:', out['class_weight_enet'])
    print('channel_mean:', out['channel_mean'])
    print('channel_std:', out['channel_std'])

    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(out, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
