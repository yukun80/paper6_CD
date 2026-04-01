_base_ = [
    '../_base_/models/snunet_c16.py',
    '../common/standard_256x256_40k_s1gfloods.py']

model = dict(
    data_preprocessor=dict(
        mean=[130.18718530288538, 130.18718530288538, 130.18718530288538] * 2,
        std=[70.85182494158705, 70.85182494158705, 70.85182494158705] * 2))
