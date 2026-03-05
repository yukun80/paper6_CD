_base_ = '../_base_/default_runtime.py'

dataset_type = 'UrbanSARFloods_CD_Dataset'
data_root = '../../datasets/urban_sar_floods_CD'
crop_size = (512, 512)

train_pipeline = [
    dict(
        type='MultiImgLoadNpyFromFile',
        expected_channels=4,
        expected_hw=crop_size),
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgApplyInvalidMask', ignore_index=255),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='horizontal'),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='vertical'),
    dict(type='MultiImgPackSegInputs')
]
test_pipeline = [
    dict(
        type='MultiImgLoadNpyFromFile',
        expected_channels=4,
        expected_hw=crop_size),
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgApplyInvalidMask', ignore_index=255),
    dict(type='MultiImgPackSegInputs')
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='meta/train_list.txt',
        data_prefix=dict(
            seg_map_path='train/label',
            img_path_from='train/A_npy',
            img_path_to='train/B_npy'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='meta/val_list.txt',
        data_prefix=dict(
            seg_map_path='val/label',
            img_path_from='val/A_npy',
            img_path_to='val/B_npy'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='mmseg.IoUMetric', iou_metrics=['mFscore', 'mIoU'], prefix='val')
test_evaluator = dict(
    type='mmseg.IoUMetric', iou_metrics=['mFscore', 'mIoU'], prefix='test')

norm_cfg = dict(type='BN', requires_grad=True)
data_preprocessor = dict(
    type='DualInputSegDataPreProcessor',
    mean=[
        0.23651549, 0.31761484, -14.57879175, -8.6098158,
        0.18514981, 0.26901252, -14.29073382, -8.33534564
    ],
    std=[
        0.16280619, 0.20849304, 4.07141682, 3.94773216,
        0.14008107, 0.19767644, 4.21006244, 4.05494136
    ],
    bgr_to_rgb=False,
    size_divisor=32,
    pad_val=0,
    seg_pad_val=255,
    test_cfg=dict(size_divisor=32))

base_channels = 16
model = dict(
    type='DIEncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='FC_Siam_diff',
        in_channels=4,
        base_channel=base_channels),
    neck=None,
    decode_head=dict(
        type='mmseg.FCNHead',
        in_channels=base_channels,
        channels=base_channels,
        in_index=-1,
        num_convs=0,
        concat_input=False,
        dropout_ratio=0,
        num_classes=3,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(
                type='mmseg.CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0),
            dict(
                type='mmseg.DiceLoss',
                use_sigmoid=False,
                activate=True,
                ignore_index=255,
                loss_weight=0.5)
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
    backbone_inchannels=4)

optimizer = dict(
    type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, _scope_='mmengine')
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=False,
        begin=0,
        end=2000,
        _scope_='mmengine'),
    dict(
        type='PolyLR',
        power=0.9,
        begin=2000,
        end=90000,
        eta_min=0.0,
        by_epoch=False,
        _scope_='mmengine')
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=90000, val_interval=2000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=2000,
        save_best='val/mIoU',
        rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='CDVisualizationHook', draw=False))
