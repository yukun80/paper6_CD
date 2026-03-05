# dataset settings
dataset_type = 'WCDataset'
data_root = '../../datasets/urban_sar_floods_CD'
classes = ('non_flood', 'flood_class_1', 'flood_class_2')
palette = ((0, 0, 0), (255, 0, 0), (0, 255, 0))

img_scale = (512, 512)

# Loaded from datasets/urban_sar_floods_CD/meta/data_norm.txt.
channel_mean = [
    0.23651549,
    0.31761484,
    0.18514981,
    0.26901252,
    -14.57879175,
    -8.60981580,
    -14.29073382,
    -8.33534564,
]
channel_std = [
    0.16280619,
    0.20849304,
    0.14008107,
    0.19767644,
    4.07141682,
    3.94773216,
    4.21006244,
    4.05494136,
]

train_pipeline = [
    dict(type='wc_LoadNpyFromFile', allow_nan=False, nan_fill_value=0.0),
    dict(type='wc_StackByChannel', keys=('img', 'aux')),
    dict(type='LoadAnnotations'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomRotate', prob=0.5, degree=90),
    dict(type='wc_MaskInvalidPixels', ignore_index=255),
    dict(type='wc_Standardize', mean=channel_mean, std=channel_std),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]

test_pipeline = [
    dict(type='wc_LoadNpyFromFile', allow_nan=False, nan_fill_value=0.0),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='wc_StackByChannel', keys=('img', 'aux')),
            dict(type='wc_Standardize', mean=channel_mean, std=channel_std),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train/A_npy',
        img_suffix='.npy',
        aux_dir='train/B_npy',
        aux_suffix='.npy',
        ann_dir='train/label',
        seg_map_suffix='.tif',
        split='meta/train_list.txt',
        pipeline=train_pipeline,
        classes=classes,
        palette=palette,
        reduce_zero_label=False),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='val/A_npy',
        img_suffix='.npy',
        aux_dir='val/B_npy',
        aux_suffix='.npy',
        ann_dir='val/label',
        seg_map_suffix='.tif',
        split='meta/val_list.txt',
        pipeline=test_pipeline,
        classes=classes,
        palette=palette,
        reduce_zero_label=False),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='val/A_npy',
        img_suffix='.npy',
        aux_dir='val/B_npy',
        aux_suffix='.npy',
        ann_dir='val/label',
        seg_map_suffix='.tif',
        split='meta/val_list.txt',
        pipeline=test_pipeline,
        classes=classes,
        palette=palette,
        reduce_zero_label=False))
