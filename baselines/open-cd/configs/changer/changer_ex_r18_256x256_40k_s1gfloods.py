_base_ = [
    '../_base_/models/changer_r18.py',
    '../common/standard_256x256_40k_s1gfloods.py']

crop_size = (256, 256)
model = dict(
    data_preprocessor=dict(
        mean=[130.18718530288538, 130.18718530288538, 130.18718530288538] * 2,
        std=[70.85182494158705, 70.85182494158705, 70.85182494158705] * 2),
    backbone=dict(
        interaction_cfg=(
            None,
            dict(type='SpatialExchange', p=1 / 2),
            dict(type='ChannelExchange', p=1 / 2),
            dict(type='ChannelExchange', p=1 / 2))),
    decode_head=dict(
        num_classes=2,
        sampler=dict(type='mmseg.OHEMPixelSampler', thresh=0.7, min_kept=100000)))

train_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgRandomRotFlip', rotate_prob=0.5, flip_prob=0.5, degree=(-20, 20)),
    dict(type='MultiImgRandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='MultiImgExchangeTime', prob=0.5),
    dict(
        type='MultiImgPhotoMetricDistortion',
        brightness_delta=10,
        contrast_range=(0.8, 1.2),
        saturation_range=(0.8, 1.2),
        hue_delta=10),
    dict(type='MultiImgPackSegInputs')
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

optimizer = dict(
    type='AdamW', lr=0.005, betas=(0.9, 0.999), weight_decay=0.05)
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=optimizer)
