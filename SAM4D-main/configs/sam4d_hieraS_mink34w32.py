_env = dict(
    sam4d_filter_cfg={
        'score': 0.3,
        'volume': 10,
        'semantic-black_list': ['road', 'sidewalk', 'crosswalk', 'curb', 'lane line'],
    },

    img_cfg=dict(
        image_shape=[1, 3, 768, 768],  # 4 cameras, 3 channels, h 512, w 512
        rot_lim=[-25.0, 25.0],  # degree
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225],
        cam_channels=[  # mixed dataset may have different camera channels
            # ['V4L_F0', 'CAMERA_PANO_FRONT_resize', 'CAMERA_PANO_FRONT'],
            # ['V4L_L0', 'CAMERA_PANO_LEFT_resize', 'CAMERA_PANO_LEFT'],
            # ['V4L_R0', 'CAMERA_PANO_RIGHT_resize', 'CAMERA_PANO_RIGHT'],
            # ['V4L_B0', 'CAMERA_PANO_BACK_resize', 'CAMERA_PANO_BACK'],
            ['CAM_HK_FS0_L', 'CAMERA_FRONT_resize', 'CAMERA_FRONT', 'FRONT', 'CAM_FRONT'],
            # ['CAMERA_LEFT_FRONT', 'FRONT_LEFT'],
            # ['CAMERA_LEFT_BACK', 'SIDE_LEFT'],
            # ['CAMERA_RIGHT_FRONT', 'FRONT_RIGHT'],
            # ['CAMERA_RIGHT_BACK', 'SIDE_RIGHT'],
        ],
        cam2image_linear=[True],  # [True, True, True, True, True],
    ),

    max_num_objects=3,
    num_maskmem=7,
    feat_dim=256,
    mem_dim=256,

    image_size=768,
    img_backbone='hiera_s',

    voxel_size=0.05,  # used in datapipeline SparseVoxelization to downsample points
    post_voxel_size=0.15,  # used in backbone MinkUNetBackboneV2
    pc_channels=4,
    point_backbone='MinkUNet34-W32',  # 'MinkUNet18-W16', MinkUNet18-W32, MinkUNet34-W32, MinkUNet50-W64, MinkUNet101-W64 ...
    use_xyz_feat=False,
    lss_dbound=[1, 77.8, 0.6],  # 128 points

    use_high_res_features_in_sam=True,
    iou_prediction_use_sigmoid=True,
    pred_obj_scores=True,
    pred_obj_scores_mlp=True,
    fixed_no_obj_ptr=True,
    use_multimask_token_for_obj_ptr=True,
    use_obj_ptrs_in_encoder=True,
    proj_tpos_enc_in_obj_ptrs=True,
    use_mlp_for_obj_ptr_proj=True,

)

__modal = dict(
    img=dict(
        image_size=_env['image_size'],
        backbone_stride=16,
        pe_cfg=dict(
            num_pos_feats=_env['feat_dim'],
            normalize=True,
            scale=None,
            temperature=10000,
        ),
    ),
    pts=dict(
        use_xyz_feat=_env['use_xyz_feat'],
        pe_cfg=dict(
            num_pos_feats=_env['feat_dim'],
        ),
        union_pe_cfg=dict(
            num_pos_feats=_env['feat_dim'],
            lss_dbound=_env['lss_dbound'],
            image_size=_env['image_size'],
            backbone_stride=16,
            add_layernorm=False,
        ),
    ),
)

## image backbone cfg
if _env['img_backbone'] == 'hiera_l':
    img_bkb_cfg = dict(
        bkb=dict(
            embed_dim=144,
            num_heads=2,
            stages=[2, 6, 36, 4],
            global_att_blocks=[23, 33, 43],
            window_pos_embed_bkg_spatial_size=[7, 7],
            window_spec=[8, 4, 16, 8],
        ),
        bkb_channel_list=[1152, 576, 288, 144],
    )
elif _env['img_backbone'] == 'hiera_b+':
    img_bkb_cfg = dict(
        bkb=dict(
            embed_dim=112,
            num_heads=2,
        ),
        bkb_channel_list=[896, 448, 224, 112],
    )
elif _env['img_backbone'] == 'hiera_s':
    img_bkb_cfg = dict(
        bkb=dict(
            embed_dim=96,
            num_heads=1,
            stages=[1, 2, 11, 2],
            global_att_blocks=[7, 10, 13],
            window_pos_embed_bkg_spatial_size=[7, 7],
        ),
        bkb_channel_list=[768, 384, 192, 96],
    )
elif _env['img_backbone'] == 'hiera_t':
    img_bkb_cfg = dict(
        bkb=dict(
            embed_dim=96,
            num_heads=1,
            stages=[1, 2, 7, 2],
            global_att_blocks=[5, 7, 9],
            window_pos_embed_bkg_spatial_size=[7, 7],
        ),
        bkb_channel_list=[768, 384, 192, 96],
    )
else:
    raise NotImplementedError

__mink_base_channels = int(_env['point_backbone'].split('-')[1].lstrip('W'))
__mink_encoder_channels = [int(x * __mink_base_channels / 32) for x in [32, 64, 128]] + [_env['feat_dim']]  # stride to 16
__mink_decoder_channels = [_env['feat_dim'], _env['feat_dim'] // 2]  # stride up to 4
__mink_encoder_depth = int(_env['point_backbone'].split('-')[0].lstrip('MinkUNet'))
assert __mink_encoder_depth in [18, 34, 50, 101, 152, 200], \
    f'MinkUNet depth must be 18, 34, 50, 101, 152, but got {__mink_encoder_depth}'
__point_backbone = dict(
    type='MinkUNetBackboneV2',
    in_channels=_env['pc_channels'] if _env['use_xyz_feat'] else _env['pc_channels'] - 3,
    base_channels=__mink_base_channels,
    num_stages=4,
    encoder_channels=__mink_encoder_channels,
    encoder_depth=__mink_encoder_depth,
    decoder_channels=__mink_decoder_channels,
    decoder_blocks=[2, 2, 2],
    pres=_env['voxel_size'],
    vres=_env['post_voxel_size'],
)

model = dict(
    type='SAM4DBase',

    ####### Training specific params #######
    # box/point input and corrections
    prob_to_use_pt_input_for_train=0.5,
    prob_to_use_pt_input_for_eval=0.0,
    prob_to_use_box_input_for_train=0.5,  # 0.5*0.5 = 0.25 prob to use box instead of points
    prob_to_use_box_input_for_eval=0.0,
    prob_to_sample_from_gt_for_train=0.1,  # with a small prob, sampling correction points from GT mask instead of prediction errors
    num_frames_to_correct_for_train=2,  # iteratively sample on random 1~2 frames (always include the first frame)
    num_frames_to_correct_for_eval=1,  # only iteratively sample on first frame
    rand_frames_to_correct_for_train=True,  # random #init-cond-frame ~ 2
    add_all_frames_to_correct_as_cond=True,
    # when a frame receives a correction click, it becomes a conditioning frame (even if it's not initially a conditioning frame)
    # maximum 2 initial conditioning frames
    num_init_cond_frames_for_train=1,
    rand_init_cond_frames_for_train=True,  # random 1~2
    num_correction_pt_per_frame=7,
    use_act_ckpt_iterative_pt_sampling=False,

    ####### Basic params #######
    feat_dim=_env['feat_dim'],
    mem_dim=_env['mem_dim'],
    num_maskmem=_env['num_maskmem'],
    max_obj_ptrs_in_encoder=_env['num_maskmem'] + 1,
    # image_size=_env['image_size'],
    modal=__modal,
    # apply scaled sigmoid on mask logits for memory encoder, and directly feed input mask as output mask
    sigmoid_scale_for_mem_enc=20.0,
    sigmoid_bias_for_mem_enc=-10.0,
    binarize_mask_from_pts_for_mem_enc=False,
    use_mask_input_as_output_without_sam=True,
    # Memory
    directly_add_no_mem_embed=True,
    no_obj_embed_spatial=True,
    # use high-resolution feature map in the SAM mask decoder
    use_high_res_features_in_sam=_env['use_high_res_features_in_sam'],
    # output 3 masks on the first click on initial conditioning frames
    multimask_output_in_sam=True,
    # SAM heads,
    iou_prediction_use_sigmoid=_env['iou_prediction_use_sigmoid'],
    # cross-attend to object pointers from other frames (based on SAM output tokens) in the encoder
    use_obj_ptrs_in_encoder=_env['use_obj_ptrs_in_encoder'],
    add_tpos_enc_to_obj_ptrs=True,
    proj_tpos_enc_in_obj_ptrs=_env['proj_tpos_enc_in_obj_ptrs'],
    use_signed_tpos_enc_to_obj_ptrs=True,
    only_obj_ptrs_in_the_past_for_eval=True,
    # object occlusion prediction
    pred_obj_scores=_env['pred_obj_scores'],
    pred_obj_scores_mlp=_env['pred_obj_scores_mlp'],
    fixed_no_obj_ptr=_env['fixed_no_obj_ptr'],
    # multimask tracking settings
    multimask_output_for_tracking=True,
    use_multimask_token_for_obj_ptr=_env['use_multimask_token_for_obj_ptr'],
    multimask_min_pt_num=0,
    multimask_max_pt_num=1,
    use_mlp_for_obj_ptr_proj=_env['use_mlp_for_obj_ptr_proj'],
    # Compilation flag
    compile_image_encoder=False,
    # Memory pe
    encode_ego_pose_in_mem=True,
    use_ego_pose_in_pe=True,

    ## 1. voxelization (done in data pipeline)
    voxelization=None,

    ## 2. backbone (optional)
    backbone=dict(
        point=dict(
            sparse_3d=[
                __point_backbone,
            ],
        ),
        image=[
            dict(
                type='Hiera',
                **img_bkb_cfg['bkb'],
            ),
            dict(
                type='FpnNeck',
                scalp=1,
                d_model=_env['feat_dim'],
                backbone_channel_list=img_bkb_cfg['bkb_channel_list'],
                fpn_top_down_levels=[2, 3],  # output level 0 and 1 directly use the backbone features
                fpn_interp_model='nearest',
            ),
        ],
    ),

    ## 3. fusion (optional)
    fusion=dict(
        type='SAM4DFusion',
        memory_attention=dict(
            type='MemoryAttention',
            d_model=256,
            pos_enc_at_input=True,
            layer=dict(
                # _target_='MemoryAttentionLayer',
                activation='relu',
                dim_feedforward=2048,
                dropout=0.1,
                pos_enc_at_attn=False,
                self_attention=dict(
                    # _target_='Attention',
                    use_rope=False,
                    embedding_dim=_env['feat_dim'],
                    num_heads=1,
                    downsample_rate=1,
                    dropout=0.1,
                ),
                d_model=256,
                pos_enc_at_cross_attn_keys=True,
                pos_enc_at_cross_attn_queries=False,
                cross_attention=dict(
                    # _target_='Attention',
                    use_rope=False,
                    embedding_dim=_env['feat_dim'],
                    num_heads=1,
                    downsample_rate=1,
                    dropout=0.1,
                    kv_in_dim=_env['mem_dim'],
                ),
            ),
            num_layers=4,
        ),
        memory_encoder=dict(
            image=dict(
                type='MemoryEncoder',
                in_dim=_env['feat_dim'],
                out_dim=_env['mem_dim'],
                position_encoding=None,
                mask_downsampler=dict(
                    # _target_='MaskDownSampler',
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                fuser=dict(
                    # _target_='Fuser',
                    layer=dict(
                        # _target_='CXBlock',
                        dim=256,
                        kernel_size=7,
                        padding=3,
                        layer_scale_init_value=1e-6,
                        use_dwconv=True  # depth-wise convs
                    ),
                    num_layers=2,
                ),
            ),
            point=dict(
                type='PtsMemoryEncoder',
                in_dim=_env['feat_dim'],
                out_dim=_env['mem_dim'],
                # position_encoding=dict(
                #     # _target_='PositionEmbeddingSine',
                #     num_pos_feats=_env['mem_dim'],
                # ),
                mask_downsampler=dict(
                    # _target_='MaskDownSampler',
                    kernel_size=2,
                    stride=2,
                    padding=0,
                    total_stride=16,
                ),
                fuser=dict(
                    # _target_='Fuser',
                    layer=dict(
                        dim=256,
                        kernel_size=7,
                        padding=0,
                        layer_scale_init_value=1e-6,
                    ),
                    num_layers=2,
                ),
            ),
        ),
    ),

    ## 4. head
    head=dict(
        sam=dict(
            type='SAMHead',
            # before task
            before_task=None,
            # head
            modal=__modal,
            num_multimask_outputs=3,
            transformer=dict(
                # _target_='TwoWayTransformer',
                depth=2,
                embedding_dim=_env['feat_dim'],
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=_env['feat_dim'],
            mem_dim=_env['mem_dim'],
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=_env['use_high_res_features_in_sam'],
            iou_prediction_use_sigmoid=_env['iou_prediction_use_sigmoid'],
            pred_obj_scores=_env['pred_obj_scores'],
            pred_obj_scores_mlp=_env['pred_obj_scores_mlp'],
            use_multimask_token_for_obj_ptr=_env['use_multimask_token_for_obj_ptr'],
            use_obj_ptrs_in_encoder=_env['use_obj_ptrs_in_encoder'],
            proj_tpos_enc_in_obj_ptrs=_env['proj_tpos_enc_in_obj_ptrs'],
            use_mlp_for_obj_ptr_proj=_env['use_mlp_for_obj_ptr_proj'],
            fixed_no_obj_ptr=_env['fixed_no_obj_ptr'],
            sam_mask_decoder_extra_args={},
        ),
    ),

)

data = dict(
    test=dict(
        pipeline=[
            dict(type='LoadPointsFromFile'),
            dict(type='LoadImageFromFile'),
            dict(type='ImageResize', final_dim=[768, 768]),
            dict(type='ImageNormalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            dict(type='SparseVoxelization', voxel_size=_env['voxel_size']),
        ],
    ),
)
