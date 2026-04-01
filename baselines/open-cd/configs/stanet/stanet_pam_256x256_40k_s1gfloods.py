_base_ = [
    '../_base_/models/stanet_r18.py',
    '../common/standard_256x256_40k_s1gfloods.py']

crop_size = (256, 256)
model = dict(
    pretrained=None,
    data_preprocessor=dict(
        mean=[130.18718530288538, 130.18718530288538, 130.18718530288538] * 2,
        std=[70.85182494158705, 70.85182494158705, 70.85182494158705] * 2),
    decode_head=dict(sa_mode='PAM'),
    test_cfg=dict(
        mode='slide',
        crop_size=crop_size,
        stride=(crop_size[0] // 2, crop_size[1] // 2)))
