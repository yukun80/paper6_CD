_base_ = [
    './base/urban_sar_floods_dataset_3c.py',
    './base/scheduler_30e.py',
    './base/default_runtime.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)

# class weights derived from datasets/urban_sar_floods_CD/meta/data_norm.txt
# num_0 : 1161425968, num_1 : 18120626, num_2 : 363550
# w = 1 / log(1.02 + freq)
class_weight = (1.43820336, 28.77954135, 49.73971985)

model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='CMCD',
        enc_opt_dims=[64, 256, 512, 1024, 2048],
        opt_in_channels=4,
        backbone_opt_cfg=dict(
            type='TIMMBackbone',
            model_name='resnet50',
            in_channels=4,
            out_indices=(0, 1, 2, 3, 4),
            output_stride=32,
            pretrained=False),
        enc_sar_dims=[16, 24, 48, 120, 352],
        sar_in_channels=4,
        backbone_sar_cfg=dict(
            type='TIMMBackbone',
            model_name='efficientnet_b2',
            in_channels=4,
            out_indices=(0, 1, 2, 3, 4),
            pretrained=False),
        center_block='dblock',
        side_dim=64,
        norm_cfg=norm_cfg),
    decode_head=dict(
        type='FCNHead',
        in_channels=64,
        in_index=0,
        channels=32,
        num_convs=2,
        concat_input=False,
        dropout_ratio=0,
        num_classes=3,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_name='loss_ce',
                loss_weight=1.0,
                class_weight=class_weight),
            dict(
                type='DiceLoss',
                loss_name='loss_dice',
                loss_weight=1.0,
                class_weight=class_weight)
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
