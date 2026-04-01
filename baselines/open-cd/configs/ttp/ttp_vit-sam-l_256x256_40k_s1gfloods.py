_base_ = [
    '../_base_/models/ttp_vit-sam-l.py',
    '../common/standard_256x256_40k_s1gfloods.py']

crop_size = (256, 256)
feature_size = (64, 64)

model = dict(
    data_preprocessor=dict(
        mean=[130.18718530288538, 130.18718530288538, 130.18718530288538] * 2,
        std=[70.85182494158705, 70.85182494158705, 70.85182494158705] * 2),
    backbone=dict(
        encoder_cfg=dict(img_size=crop_size)),
    decode_head=dict(out_size=feature_size),
    test_cfg=dict(mode='whole'))

train_dataloader = dict(batch_size=2)

optimizer = dict(
    type='AdamW', lr=0.0004, betas=(0.9, 0.999), weight_decay=0.05)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=optimizer)
