_base_ = [
    '../_base_/models/cgnet.py',
    '../common/standard_256x256_40k_s1gfloods.py']

crop_size = (256, 256)
model = dict(
    data_preprocessor=dict(
        mean=[130.18718530288538, 130.18718530288538, 130.18718530288538] * 2,
        std=[70.85182494158705, 70.85182494158705, 70.85182494158705] * 2),
    backbone=dict(pretrained=False),
    test_cfg=dict(
        mode='slide',
        crop_size=crop_size,
        stride=(crop_size[0] // 2, crop_size[1] // 2)))

optimizer = dict(
    type='AdamW',
    lr=5e-4,
    betas=(0.9, 0.999),
    weight_decay=0.0025)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=optimizer)
