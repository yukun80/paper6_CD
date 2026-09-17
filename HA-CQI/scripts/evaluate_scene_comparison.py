#!/usr/bin/env python3
"""统一评估当前保存的整景二值图（可能包含人工更新），不读取切片或概率。

conda activate hacqi
python HA-CQI/scripts/evaluate_scene_comparison.py --dry-run
python HA-CQI/scripts/evaluate_scene_comparison.py
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
import hashlib
import os
from pathlib import Path
import tempfile

import numpy as np
import rasterio
from rasterio.windows import Window

# 配置区：所有相对路径均以仓库为基准；模型列表明确固定，不自动发现旧模型。
REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = 'HA-CQI/outputs'
OPENCD_ROOT = 'baselines/open-cd/outputs'
HARMO_ROOT = 'HA-CQI/outputs/HA-CQI-OSCD-260912'
REGIONS = {
    'Zhengzhou': 'datasets/GF3_Henan/GF3_ZhengzhouC_label.tif',
    'Zhuozhou': 'datasets/GF3_Zhuozhou/GF3_Zhuozhou_label.tif',
    'Guangxi': 'datasets/LT1_Guangxi/LT_Guangxi_label.tif',
}
MODELS = (
    ('FC-Siam-Diff', 'fc_siam_diff_256x256_40k_s1gfloods'),
    ('BIT', 'bit_r18_256x256_40k_s1gfloods'),
    ('LightCDNet', 'lightcdnet_s_256x256_40k_s1gfloods'),
    ('ChangeFormer', 'changeformer_mit-b0_256x256_40k_s1gfloods'),
    ('CGNet', 'cgnet_256x256_40k_s1gfloods'),
    ('SNUNet', 'snunet_c16_256x256_40k_s1gfloods'),
    ('HANet', 'hanet_256x256_40k_s1gfloods'),
    ('TTP', 'ttp_vit-sam-b_256x256_40k_s1gfloods'),
    ('ChangeDINO', 'changedino'),
    ('HarmoSSM', 'HA-CQI-OSCD-260912'),
)
FIELDS = ('region model model_tag TP TN FP FN IoU Precision Recall F1 OA '
          'valid_pixels excluded_pixels mask_policy label_path prediction_path '
          'label_sha256 prediction_sha256').split()
MASK_POLICY = 'label_and_all_models_valid_intersection'


def resolve(path: str | Path) -> Path:
    return (REPO_ROOT / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_grid(reference, image) -> None:
    """按整景四角的像素位移判断网格一致性，容忍浮点舍入误差。"""
    if image.count != 1 or reference.count != 1:
        raise ValueError(f'Single band required: {image.name}')
    if image.shape != reference.shape or image.crs != reference.crs or image.crs is None:
        raise ValueError(f'Shape/CRS mismatch or missing CRS: {image.name}')
    mapping = ~reference.transform * image.transform
    for x, y in ((0, 0), (image.width, 0), (0, image.height),
                 (image.width, image.height)):
        px, py = mapping * (x, y)
        if not np.isfinite([px, py]).all() or max(abs(px-x), abs(py-y)) > 1e-6:
            raise ValueError(f'Grid mismatch: {image.name}')


def read_binary(image, window: Window, nodata: int):
    """先严格校验编码，再排除显式 NoData 与 GDAL 掩膜。"""
    if image.nodata is not None and image.nodata != nodata:
        raise ValueError(f'Unexpected NoData {image.nodata}: {image.name}')
    values = image.read(1, window=window)
    if not np.isin(values, [0, 1, nodata]).all():
        raise ValueError(f'Unknown binary values: {image.name}')
    valid = (values != nodata) & (image.read_masks(1, window=window) > 0)
    return values == 1, valid


def metrics(tp: int, tn: int, fp: int, fn: int) -> dict:
    def divide(a, b):
        return a / b if b else float('nan')
    return dict(IoU=divide(tp, tp+fp+fn), Precision=divide(tp, tp+fp),
                Recall=divide(tp, tp+fn), F1=divide(2*tp, 2*tp+fp+fn),
                OA=divide(tp+tn, tp+tn+fp+fn))


def evaluate_region(region: str, label: Path, predictions: list,
                    block_size: int = 512) -> list[dict]:
    """分块遍历整景一次，各模型在同一有效区域累计混淆矩阵。"""
    if block_size < 1 or not predictions:
        raise ValueError('Positive block size and nonempty model list required')
    paths = [label] + [p for _, _, p in predictions]
    before = {p: sha256(p) for p in paths}
    counts = np.zeros((len(predictions), 4), dtype=np.int64)
    valid_pixels = 0
    with ExitStack() as stack:
        gt = stack.enter_context(rasterio.open(label))
        images = [stack.enter_context(rasterio.open(p)) for _, _, p in predictions]
        validate_grid(gt, gt)
        for image in images:
            validate_grid(gt, image)
        total = gt.width * gt.height
        for top in range(0, gt.height, block_size):
            for left in range(0, gt.width, block_size):
                window = Window(left, top, min(block_size, gt.width-left),
                                min(block_size, gt.height-top))
                truth, common = read_binary(gt, window, 3)
                blocks = []
                for image in images:
                    pred, valid = read_binary(image, window, 255)
                    blocks.append(pred)
                    common &= valid
                valid_pixels += int(common.sum())
                for index, pred in enumerate(blocks):
                    counts[index] += [np.count_nonzero(common & pred & truth),
                                      np.count_nonzero(common & ~pred & ~truth),
                                      np.count_nonzero(common & pred & ~truth),
                                      np.count_nonzero(common & ~pred & truth)]
    if not valid_pixels:
        raise ValueError(f'{region}: empty common valid area')
    for path in paths:
        if sha256(path) != before[path]:
            raise ValueError(f'Input changed during evaluation: {path}')
    rows = []
    for (name, tag, path), confusion in zip(predictions, counts):
        tp, tn, fp, fn = map(int, confusion)
        assert tp+tn+fp+fn == valid_pixels
        rows.append(dict(region=region, model=name, model_tag=tag,
                         TP=tp, TN=tn, FP=fp, FN=fn, **metrics(tp, tn, fp, fn),
                         valid_pixels=valid_pixels, excluded_pixels=total-valid_pixels,
                         mask_policy=MASK_POLICY, label_path=str(label),
                         prediction_path=str(path), label_sha256=before[label],
                         prediction_sha256=before[path]))
    return rows


def write_tsv(path: Path, rows: list[dict]) -> None:
    """完整写入临时文件后替换目标，避免留下部分 TSV。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False,
                                         encoding='utf-8', newline='') as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter='\t')
            writer.writeheader()
            for row in rows:
                writer.writerow({k: 'NaN' if isinstance(v, float) and np.isnan(v)
                                 else v for k, v in row.items()})
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--regions', nargs='+', choices=list(REGIONS), default=list(REGIONS))
    parser.add_argument('--output-dir', default=OUTPUT_DIR)
    parser.add_argument('--dry-run', action='store_true', help='完整预检及统计，不写文件')
    args = parser.parse_args(argv)
    try:
        results = {}
        for region in dict.fromkeys(args.regions):
            predictions = [(name, tag, resolve(HARMO_ROOT if name == 'HarmoSSM'
                           else Path(OPENCD_ROOT)/tag)/region/'mosaic/change_binary.tif')
                           for name, tag in MODELS]
            results[region] = evaluate_region(region, resolve(REGIONS[region]), predictions)
            print(f'{region}: models={len(results[region])}, '
                  f'common_valid={results[region][0]["valid_pixels"]}', flush=True)
        # 全部地区成功后才允许写正式表，任何缺失模型都不会被静默移除。
        if not args.dry_run:
            for region, rows in results.items():
                target = resolve(args.output_dir)/f'scene_metrics_{region}.tsv'
                write_tsv(target, rows)
                print(f'Saved: {target}', flush=True)
        return 0
    except (OSError, ValueError, rasterio.errors.RasterioError) as exc:
        print(f'ERROR: {exc}', flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
