import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsparse.nn as spnn
import torchsparse.nn.functional as spf
from mmengine import MODELS
from torch.nn.init import trunc_normal_
from torchsparse.tensor import PointTensor, SparseTensor

from sam4d.misc import get_1d_sine_pe, select_closest_cond_frames
from sam4d.position_encoding import UnionPositionEmbedding
from sam4d.utils import PoseEncoder
from .model_template import ModelTemplate

# a large negative value as a placeholder score for missing objects
NO_OBJ_SCORE = -1024.0


@MODELS.register_module()
class SAM4DBase(nn.Module):
    def __init__(
            self,
            voxelization,
            backbone,  # include image and lidar backbone
            fusion,  # include memory attention and memory encoder
            head,
            feat_dim=256,
            mem_dim=256,
            num_maskmem=7,  # default 1 input frame + 6 previous frames
            modal=dict(img=dict(image_size=1024, backbone_stride=16)),
            # image_size=512,
            # backbone_stride=16,  # stride of the image backbone output
            sigmoid_scale_for_mem_enc=1.0,  # scale factor for mask sigmoid prob
            sigmoid_bias_for_mem_enc=0.0,  # bias factor for mask sigmoid prob
            # During evaluation, whether to binarize the sigmoid mask logits on interacted frames with clicks
            binarize_mask_from_pts_for_mem_enc=False,
            use_mask_input_as_output_without_sam=False,
            # on frames with mask input, whether to directly output the input mask without using a SAM prompt encoder + mask decoder
            # The maximum number of conditioning frames to participate in the memory attention (-1 means no limit; if there are more conditioning frames than this limit,
            # we only cross-attend to the temporally closest `max_cond_frames_in_attn` conditioning frames in the encoder when tracking each frame). This gives the model
            # a temporal locality when handling a large number of annotated frames (since closer frames should be more important) and also avoids GPU OOM.
            max_cond_frames_in_attn=-1,
            # on the first frame, whether to directly add the no-memory embedding to the image feature
            # (instead of using the transformer encoder)
            directly_add_no_mem_embed=False,
            # whether to use high-resolution feature maps in the SAM mask decoder
            use_high_res_features_in_sam=False,
            # whether to output multiple (3) masks for the first click on initial conditioning frames
            multimask_output_in_sam=False,
            # the minimum and maximum number of clicks to use multimask_output_in_sam (only relevant when `multimask_output_in_sam=True`;
            # default is 1 for both, meaning that only the first click gives multimask output; also note that a box counts as two points)
            multimask_min_pt_num=1,
            multimask_max_pt_num=1,
            # whether to also use multimask output for tracking (not just for the first click on initial conditioning frames; only relevant when `multimask_output_in_sam=True`)
            multimask_output_for_tracking=False,
            # Whether to use multimask tokens for obj ptr; Only relevant when both
            # use_obj_ptrs_in_encoder=True and multimask_output_for_tracking=True
            use_multimask_token_for_obj_ptr: bool = False,
            # whether to use sigmoid to restrict ious prediction to [0-1]
            iou_prediction_use_sigmoid=False,
            # The memory bank's temporal stride during evaluation (i.e. the `r` parameter in XMem and Cutie; XMem and Cutie use r=5).
            # For r>1, the (self.num_maskmem - 1) non-conditioning memory frames consist of
            # (self.num_maskmem - 2) nearest frames from every r-th frames, plus the last frame.
            memory_temporal_stride_for_eval=1,
            # whether to apply non-overlapping constraints on the object masks in the memory encoder during evaluation (to avoid/alleviate superposing masks)
            non_overlap_masks_for_mem_enc=False,
            # whether to cross-attend to object pointers from other frames (based on SAM output tokens) in the encoder
            use_obj_ptrs_in_encoder=False,
            # the maximum number of object pointers from other frames in encoder cross attention (only relevant when `use_obj_ptrs_in_encoder=True`)
            max_obj_ptrs_in_encoder=16,
            # whether to add temporal positional encoding to the object pointers in the encoder (only relevant when `use_obj_ptrs_in_encoder=True`)
            add_tpos_enc_to_obj_ptrs=True,
            # whether to add an extra linear projection layer for the temporal positional encoding in the object pointers to avoid potential interference
            # with spatial positional encoding (only relevant when both `use_obj_ptrs_in_encoder=True` and `add_tpos_enc_to_obj_ptrs=True`)
            proj_tpos_enc_in_obj_ptrs=False,
            # whether to use signed distance (instead of unsigned absolute distance) in the temporal positional encoding in the object pointers
            # (only relevant when both `use_obj_ptrs_in_encoder=True` and `add_tpos_enc_to_obj_ptrs=True`)
            use_signed_tpos_enc_to_obj_ptrs=False,
            # whether to only attend to object pointers in the past (before the current frame) in the encoder during evaluation
            # (only relevant when `use_obj_ptrs_in_encoder=True`; this might avoid pointer information too far in the future to distract the initial tracking)
            only_obj_ptrs_in_the_past_for_eval=False,
            # Whether to predict if there is an object in the frame
            pred_obj_scores: bool = False,
            # Whether to use an MLP to predict object scores
            pred_obj_scores_mlp: bool = False,
            # Only relevant if pred_obj_scores=True and use_obj_ptrs_in_encoder=True;
            # Whether to have a fixed no obj pointer when there is no object present
            # or to use it as an additive embedding with obj_ptr produced by decoder
            fixed_no_obj_ptr: bool = False,
            # Soft no object, i.e. mix in no_obj_ptr softly,
            # hope to make recovery easier if there is a mistake and mitigate accumulation of errors
            # soft_no_obj_ptr: bool = False,
            use_mlp_for_obj_ptr_proj: bool = False,
            # add no obj embedding to spatial frames
            no_obj_embed_spatial: bool = False,
            # extra arguments used to construct the SAM mask decoder; if not None, it should be a dict of kwargs to be passed into `MaskDecoder` class.
            # sam_mask_decoder_extra_args=None,
            compile_image_encoder: bool = False,
            # whether to encode ego pose in the memory
            encode_ego_pose_in_mem: bool = False,
            use_ego_pose_in_pe: bool = False,
            **kwargs,
    ):
        super(SAM4DBase, self).__init__()

        self.pure_model = ModelTemplate(voxelization,
                                        backbone,
                                        fusion,
                                        head)
        num_params = sum(p.numel() for p in self.pure_model.parameters() if p.requires_grad)
        print(f"===== params of pure model: {num_params / 1e6} M")
        if 'img' in modal:
            img_backbone_params = sum(p.numel() for p in self.pure_model.backbone.image.parameters() if p.requires_grad)
        else:
            img_backbone_params = 0
        if 'pts' in modal:
            pts_backbone_params = sum(p.numel() for p in self.pure_model.backbone.point.parameters() if p.requires_grad)
        else:
            pts_backbone_params = 0
        fusion_params = sum(p.numel() for p in self.pure_model.fusion.parameters() if p.requires_grad)
        head_params = sum(p.numel() for p in self.pure_model.head.parameters() if p.requires_grad)
        print(f"===== params of img backbone: {img_backbone_params / 1e6} M; pts backbone: {pts_backbone_params / 1e6} M; "
              f"fusion: {fusion_params / 1e6} M; head: {head_params / 1e6} M")

        # Part 0: union positional embedding
        self.modal = modal
        self.union_pe_func = UnionPositionEmbedding(self.modal)
        assert feat_dim == mem_dim, "due to union_pe refactor, feat_dim and mem_dim must be the same!!"

        # Part 1: the image backbone
        # Use level 0, 1, 2 for high-res setting, or just level 2 for the default setting
        self.use_high_res_features_in_sam = use_high_res_features_in_sam
        self.num_feature_levels = 3 if use_high_res_features_in_sam else 1
        self.use_obj_ptrs_in_encoder = use_obj_ptrs_in_encoder
        self.max_obj_ptrs_in_encoder = max_obj_ptrs_in_encoder
        if use_obj_ptrs_in_encoder:
            # A conv layer to downsample the mask prompt to stride 4 (the same stride as
            # low-res SAM mask logits) and to change its scales from 0~1 to SAM logit scale,
            # so that it can be fed into the SAM mask decoder to generate a pointer.
            if 'img' in self.modal:
                self.mask_downsample_img = torch.nn.Conv2d(1, 1, kernel_size=4, stride=4)
            if 'pts' in self.modal:
                self.mask_downsample_pts = spnn.Conv3d(1, 1, kernel_size=4, stride=4, bias=True)
        self.add_tpos_enc_to_obj_ptrs = add_tpos_enc_to_obj_ptrs
        if proj_tpos_enc_in_obj_ptrs:
            assert add_tpos_enc_to_obj_ptrs  # these options need to be used together
        self.proj_tpos_enc_in_obj_ptrs = proj_tpos_enc_in_obj_ptrs
        self.use_signed_tpos_enc_to_obj_ptrs = use_signed_tpos_enc_to_obj_ptrs
        self.only_obj_ptrs_in_the_past_for_eval = only_obj_ptrs_in_the_past_for_eval

        # Part 2: memory attention to condition current frame's visual features
        # with memories (and obj ptrs) from past frames
        self.hidden_dim = feat_dim

        # Part 3: memory encoder for the previous frame's outputs
        self.mem_dim = mem_dim
        self.num_maskmem = num_maskmem  # Number of memories accessible
        # Temporal encoding of the memories
        self.maskmem_tpos_enc = torch.nn.Parameter(
            torch.zeros(num_maskmem, 1, 1, self.mem_dim)
        )
        trunc_normal_(self.maskmem_tpos_enc, std=0.02)
        # a single token to indicate no memory embedding from previous frames
        self.no_mem_embed = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.no_mem_pos_enc = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        trunc_normal_(self.no_mem_embed, std=0.02)
        trunc_normal_(self.no_mem_pos_enc, std=0.02)
        self.directly_add_no_mem_embed = directly_add_no_mem_embed
        # Apply sigmoid to the output raw mask logits (to turn them from
        # range (-inf, +inf) to range (0, 1)) before feeding them into the memory encoder
        self.sigmoid_scale_for_mem_enc = sigmoid_scale_for_mem_enc
        self.sigmoid_bias_for_mem_enc = sigmoid_bias_for_mem_enc
        self.binarize_mask_from_pts_for_mem_enc = binarize_mask_from_pts_for_mem_enc
        self.non_overlap_masks_for_mem_enc = non_overlap_masks_for_mem_enc
        self.memory_temporal_stride_for_eval = memory_temporal_stride_for_eval
        # On frames with mask input, whether to directly output the input mask without
        # using a SAM prompt encoder + mask decoder
        self.use_mask_input_as_output_without_sam = use_mask_input_as_output_without_sam
        self.multimask_output_in_sam = multimask_output_in_sam
        self.multimask_min_pt_num = multimask_min_pt_num
        self.multimask_max_pt_num = multimask_max_pt_num
        self.multimask_output_for_tracking = multimask_output_for_tracking
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr
        self.iou_prediction_use_sigmoid = iou_prediction_use_sigmoid

        # Part 4: SAM-style prompt encoder (for both mask and point inputs)
        # and SAM-style mask decoder for the final mask output
        self.pred_obj_scores = pred_obj_scores
        self.pred_obj_scores_mlp = pred_obj_scores_mlp
        self.fixed_no_obj_ptr = fixed_no_obj_ptr
        # self.soft_no_obj_ptr = soft_no_obj_ptr
        if self.fixed_no_obj_ptr:
            assert self.pred_obj_scores
            assert self.use_obj_ptrs_in_encoder
        self.use_mlp_for_obj_ptr_proj = use_mlp_for_obj_ptr_proj
        self.no_obj_embed_spatial = None
        if no_obj_embed_spatial:
            self.no_obj_embed_spatial = torch.nn.Parameter(torch.zeros(1, self.mem_dim))
            trunc_normal_(self.no_obj_embed_spatial, std=0.02)

        # self._build_sam_heads()
        self.max_cond_frames_in_attn = max_cond_frames_in_attn

        # Model compilation
        if compile_image_encoder:
            # Compile the forward function (not the full module) to allow loading checkpoints.
            print(
                "Image encoder compilation is enabled. First forward pass will be slow."
            )
            self.pure_model.backbone.image.forward = torch.compile(
                self.pure_model.backbone.image.forward,
                mode="max-autotune",
                fullgraph=True,
                dynamic=False,
            )

        # ego pose pe
        # assert not (encode_ego_pose_in_mem and use_ego_pose_in_pe), f'implicit and explicit ego pose pe are mutually exclusive'
        self.encode_ego_pose_in_mem = encode_ego_pose_in_mem
        if self.encode_ego_pose_in_mem:
            self.ego_pose_encoder = PoseEncoder(self.mem_dim)

        self.use_ego_pose_in_pe = use_ego_pose_in_pe

        # head_loss.update(train_cfg=train_cfg, test_cfg=test_cfg)
        # self.head_loss = build_head_loss(head_loss)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Please use the corresponding methods in SAM2VideoPredictor for inference or SAM4DTrain for training/fine-tuning"
            "See notebooks/video_predictor_example.ipynb for an inference example."
        )

    def prepare_input_dict(self, **kwargs):
        assert 'images' in kwargs or 'points' in kwargs, f"images or points should be in kwargs"
        self.num_frames = kwargs['seq_len']

        input_dict = dict(seq_len=self.num_frames, union_pe_func=self.union_pe_func, metas=kwargs['metas'][0])
        ## ========== image ========== ##
        if 'images' in kwargs:
            assert len(kwargs['images']) == 1, f"batch size should set to 1 for SAM4D, but got {len(kwargs['images'])}"
            input_dict['images'] = kwargs['images'][0]

        ## ========== points ========== ##
        if 'points' in kwargs:
            assert len(kwargs['points']) == 1, f"batch size should set to 1 for SAM4D, but got {len(kwargs['points'])}"
            points = kwargs['points'][0]
            pt_coords = kwargs['pt_coords'][0]
            st_voxels = []
            for one_points, one_pt_coords in zip(points, pt_coords):
                st_feat = one_points if self.modal['pts']['use_xyz_feat'] else one_points[:, 3:]
                tmp_pt_coords = torch.cat((torch.zeros(one_pt_coords.shape[0], 1).to(one_pt_coords.device), one_pt_coords), dim=1).int()
                st_voxels.append(SparseTensor(st_feat, tmp_pt_coords))  # st_feat (K, C), pt_coords (K, 4) (bs, x, y, z)
            input_dict.update(dict(
                points=points,
                st_voxels=st_voxels,
            ))

        return input_dict

    def pts_sp_interpolate(self, src_coords, dst_coords, feat, src2dst_stride=1):
        pc_hash = spf.sphash(
            torch.cat([
                src_coords[:, 0].int().view(-1, 1),
                torch.floor(src_coords[:, 1:4] / src2dst_stride).int()
            ], 1))
        sparse_hash = spf.sphash(dst_coords)
        idx_query = spf.sphashquery(pc_hash, sparse_hash)
        counts = spf.spcount(idx_query.int(), len(sparse_hash))
        inserted_feat = spf.spvoxelize(feat, idx_query, counts)
        return inserted_feat

    def _use_mask_as_output(self, backbone_features, high_res_features, mask_inputs, psam_info=None, sam_info=None, meta=None):
        """
        Directly turn binary `mask_inputs` into a output mask logits without using SAM.
        (same input and output shapes as in self.pure_model.head.sam above).
        """
        assert isinstance(backbone_features, dict), f'got backbone_features type: {type(backbone_features)}'
        assert isinstance(high_res_features, dict), f'got high_res_features type: {type(high_res_features)}'
        assert isinstance(mask_inputs, dict), f'got mask_inputs type: {type(mask_inputs)}'

        keys = list(mask_inputs.keys())
        # Use -10/+10 as logits for neg/pos pixels (very close to 0/1 in prob after sigmoid).
        out_scale, out_bias = 20.0, -10.0  # sigmoid(-10.0)=4.5398e-05
        mask_inputs_float = {k: v.float() for k, v in mask_inputs.items()}
        high_res_masks = {k: v * out_scale + out_bias for k, v in mask_inputs_float.items()}
        low_res_masks = {}
        for key in keys:
            assert key in ['img', 'pts'], f'got key: {key}'
            curr_masks = high_res_masks[key]
            if key == 'img':
                low_res_masks[key] = F.interpolate(
                    curr_masks,
                    size=(curr_masks.size(-2) // 4, curr_masks.size(-1) // 4),
                    align_corners=False,
                    mode="bilinear",
                    antialias=True,  # use antialias for downsampling
                )
            else:
                assert psam_info is not None, 'psam_info is required for pts modal'
                assert curr_masks.size(1) == 1 and curr_masks.size(3) == 1, f'high_res_masks(pts).size(): {curr_masks.size()}'
                pts_C = psam_info['pts_org_feats'].C
                vxs_C = psam_info['pts_sp_tensor_info'][0]['coords']  # stride 4
                vxs_stride = psam_info['pts_sp_tensor_info'][0]['stride'][0]
                assert vxs_stride == 4, f'got vxs_stride: {vxs_stride}'
                tmp_high_res_masks = curr_masks.squeeze(1, 3).transpose(1, 0)
                tmp_low_res_masks = self.pts_sp_interpolate(pts_C, vxs_C, tmp_high_res_masks, vxs_stride)
                low_res_masks[key] = tmp_low_res_masks.transpose(1, 0).unsqueeze(1).unsqueeze(3)
                # low_res_masks[key] = SparseTensor(tmp_low_res_masks, vxs_C, stride=vxs_stride,
                #                                   spatial_range=psam_info['pts_sp_tensor_info'][0]['spatial_range'])
                # low_res_masks[key] = curr_masks

        # a dummy IoU prediction of all 1's under mask input
        ious = {k: v.new_ones(v.size(0), 1).float() for k, v in mask_inputs.items()}
        if not self.use_obj_ptrs_in_encoder:
            # all zeros as a dummy object pointer (of shape [B, C])
            obj_ptr = {k: torch.zeros(v.size(0), self.hidden_dim, device=v.device) for k, v in mask_inputs.items()}
        else:
            # produce an object pointer using the SAM decoder from the mask input
            sam_mask_inputs = {}
            for key in mask_inputs_float:
                if key == 'img':
                    sam_mask_inputs[key] = self.mask_downsample_img(mask_inputs_float[key])
                else:
                    # sam_mask_inputs[key] = low_res_masks[key]
                    pts_mask_inputs = []
                    for x in mask_inputs_float[key]:
                        assert x.size(2) == 1
                        x = SparseTensor(x.squeeze(2).transpose(1, 0), psam_info['pts_org_feats'].C.int())
                        x = self.mask_downsample_pts(x)
                        pts_mask_inputs.append(x.F.transpose(1, 0).unsqueeze(2))
                    sam_mask_inputs[key] = torch.stack(pts_mask_inputs, dim=0)
            sam_outs = self.pure_model.head.sam(
                backbone_features=backbone_features,
                mask_inputs=sam_mask_inputs,
                high_res_features=high_res_features,
                psam_info=psam_info,
                union_pe_func=self.union_pe_func,
                sam_info=sam_info,
                meta=meta,
            )
            obj_ptr = {k: v['obj_ptr'] for k, v in sam_outs.items()}
        # In this method, we are treating mask_input as output, e.g. using it directly to create spatial mem;
        # Below, we follow the same design axiom to use mask_input to decide if obj appears or not instead of relying
        # on the object_scores from the SAM decoder.
        is_obj_appearing = {k: torch.any(v.flatten(1).float() > 0.0, dim=1) for k, v in mask_inputs.items()}
        is_obj_appearing = {k: v[..., None] for k, v in is_obj_appearing.items()}
        lambda_is_obj_appearing = {k: v.float() for k, v in is_obj_appearing.items()}
        object_score_logits = {k: out_scale * v + out_bias for k, v in lambda_is_obj_appearing.items()}
        if self.pred_obj_scores:
            if self.fixed_no_obj_ptr:
                obj_ptr = {k: lambda_is_obj_appearing[k] * obj_ptr[k] for k in lambda_is_obj_appearing}
            obj_ptr = {k: obj_ptr[k] + (1 - lambda_is_obj_appearing[k]) * self.pure_model.head.sam.no_obj_ptr for k in obj_ptr}

        ret = {k: {
            "low_res_multimasks": low_res_masks[k],
            "high_res_multimasks": high_res_masks[k],
            "ious": ious[k],
            "low_res_masks": low_res_masks[k],
            "high_res_masks": high_res_masks[k],
            "obj_ptr": obj_ptr[k],
            "object_score_logits": object_score_logits[k]
        } for k in keys}

        return ret

    def forward_backbone(self, input_dict):
        """Get the image feature on the input batch."""
        ori_keys = list(input_dict.keys())

        if 'image' in self.pure_model.backbone:
            input_dict = self.pure_model.backbone.image(input_dict)  # Sequential
            if self.use_high_res_features_in_sam:
                # precompute projected level 0 and level 1 features in SAM decoder to avoid running it again on every SAM click
                input_dict["backbone_fpn"][0] = self.pure_model.head.sam.sam_mask_decoder.conv_s0(
                    input_dict["backbone_fpn"][0]
                )
                input_dict["backbone_fpn"][1] = self.pure_model.head.sam.sam_mask_decoder.conv_s1(
                    input_dict["backbone_fpn"][1]
                )
            # split image temporal dimension to list, keep it consistent with point features
            for k in ['backbone_fpn']:
                if not k in input_dict:
                    continue
                if isinstance(input_dict[k], list):
                    input_dict[k] = [[xx for xx in x] for x in input_dict[k]]
                else:
                    input_dict[k] = [x for x in input_dict[k]]

        if 'point' in self.pure_model.backbone:
            new_pts_backbone_fpn, new_pts_pos, sp_tensor_info = [], [], []
            new_pts_org_feats = []
            for t_i in range(input_dict['seq_len']):
                tmp_input_dict = dict(
                    st_voxels=input_dict['st_voxels'][t_i],
                    points=input_dict['points'][t_i],
                    union_pe_func=input_dict['union_pe_func'],
                )
                with torch.cuda.amp.autocast(enabled=False):
                    for _, module in self.pure_model.backbone['point'].items():  # ModuleDict
                        tmp_input_dict = module(tmp_input_dict)  # backbone forward
                if self.use_high_res_features_in_sam:
                    # precompute projected level 0 and level 1 features in SAM decoder to avoid running it again on every SAM click
                    tmp_input_dict["pts_backbone_fpn"][0] = self.pure_model.head.sam.sam_mask_decoder.conv_s0_pts(
                        tmp_input_dict["pts_backbone_fpn"][0]
                    )
                    tmp_input_dict["pts_backbone_fpn"][1] = self.pure_model.head.sam.sam_mask_decoder.conv_s1_pts(
                        tmp_input_dict["pts_backbone_fpn"][1]
                    )
                new_pts_org_feats.append(PointTensor(input_dict['points'][t_i], tmp_input_dict['pts_org_feats'].C))
                sp_tensor_info.append([dict(coords=f.C, stride=f.stride, spatial_range=f.spatial_range, _cache=f._caches)
                                       for f in tmp_input_dict['pts_backbone_fpn']])
                new_pts_backbone_fpn.append([f.F for f in tmp_input_dict['pts_backbone_fpn']])
                new_pts_pos.append([pos for pos in tmp_input_dict['pts_pos']])

            sp_tensor_info = [list(tup) for tup in zip(*sp_tensor_info)]  # put time dimension to the first
            new_pts_backbone_fpn = [list(tup) for tup in zip(*new_pts_backbone_fpn)]
            new_pts_pos = [list(tup) for tup in zip(*new_pts_pos)]
            #### end of each frame do spconv3d separately

            input_dict['pts_sp_tensor_info'] = sp_tensor_info
            input_dict['pts_backbone_fpn'] = new_pts_backbone_fpn
            input_dict['pts_pos'] = new_pts_pos
            input_dict['pts_org_feats'] = new_pts_org_feats

        backbone_out = {k: input_dict.pop(k) for k in list(input_dict.keys()) if k not in ori_keys}

        return backbone_out

    def _prepare_backbone_features(self, backbone_out):
        """Prepare and flatten visual features."""
        backbone_out = backbone_out.copy()
        backbone_feats, feat_sizes = {}, {}
        if "backbone_fpn" in backbone_out and backbone_out["backbone_fpn"] is not None:
            assert len(backbone_out["backbone_fpn"]) >= self.num_feature_levels

            feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels:]

            feat_sizes['img'] = [[(xx[0].shape[-2], xx[0].shape[-1]) for xx in x] for x in feature_maps]
            # flatten N, CxHxW to N, HWxC
            backbone_feats['img'] = [[xx.flatten(1).permute(1, 0) for xx in x] for x in feature_maps]

        if "pts_backbone_fpn" in backbone_out and backbone_out["pts_backbone_fpn"] is not None:
            feature_maps = backbone_out["pts_backbone_fpn"][-self.num_feature_levels:]

            feat_sizes['pts'] = [[(xx.shape[0], 1) for xx in x] for x in feature_maps]
            # points feature already in HWxC format
            backbone_feats['pts'] = feature_maps

        return backbone_out, backbone_feats, feat_sizes

    def _prepare_memory_conditioned_features(
            self,
            frame_idx,
            is_init_cond_frame,
            current_vision_feat,
            current_vision_pos_embed,
            feat_sizes,
            output_dict,
            num_frames,
            track_in_reverse=False,  # tracking in reverse time order (for demo usage)
            metas=None,
            psam_infos=None,
    ):
        """Fuse the current frame's visual feature map with previous memory."""
        keys = list(current_vision_feat.keys())
        B = current_vision_feat[keys[0]].size(1)  # batch size on this frame
        C = self.hidden_dim
        # H, W = feat_sizes[keys[0]]  # top-level (lowest-resolution) feature size
        device = current_vision_feat[keys[0]].device
        # The case of `self.num_maskmem == 0` below is primarily used for reproducing SAM on images.
        # In this case, we skip the fusion without any memory.
        if self.num_maskmem == 0:  # Disable memory and skip fusion
            pix_feat = {k: v.permute(1, 2, 0).view(B, C, *feat_sizes[k]) for k, v in current_vision_feat.items()}
            return pix_feat

        lidar_aug_matrix = torch.as_tensor(metas[frame_idx]['lidar_aug_matrix'], device=self.device)  # (4, 4)
        cur_pose = torch.as_tensor(metas[frame_idx]['pose'], device=self.device)  # (4, 4)
        num_obj_ptr_tokens = 0
        tpos_sign_mul = -1 if track_in_reverse else 1
        # Step 1: condition the visual features of the current frame on previous memories
        if not is_init_cond_frame:
            # Retrieve the memories encoded with the maskmem backbone
            to_cat_memory, to_cat_memory_pos_embed = [], []
            # Add conditioning frames's output first (all cond frames have t_pos=0 for
            # when getting temporal positional embedding below)
            assert len(output_dict["cond_frame_outputs"]) > 0
            # Select a maximum number of temporally closest cond frames for cross attention
            cond_outputs = output_dict["cond_frame_outputs"]
            selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
                frame_idx, cond_outputs, self.max_cond_frames_in_attn, max_frame_diff=self.num_maskmem + 1,
            )
            t_pos_and_prevs = [(0, out, torch.as_tensor(metas[f_id]['pose'], device=self.device), f_id)
                               for f_id, out in selected_cond_outputs.items()]
            # Add last (self.num_maskmem - 1) frames before current frame for non-conditioning memory
            # the earliest one has t_pos=1 and the latest one has t_pos=self.num_maskmem-1
            # We also allow taking the memory frame non-consecutively (with stride>1), in which case
            # we take (self.num_maskmem - 2) frames among every stride-th frames plus the last frame.
            stride = 1 if self.training else self.memory_temporal_stride_for_eval
            for t_pos in range(1, self.num_maskmem):
                t_rel = self.num_maskmem - t_pos  # how many frames before current frame
                if t_rel == 1:
                    # for t_rel == 1, we take the last frame (regardless of r)
                    if not track_in_reverse:
                        # the frame immediately before this frame (i.e. frame_idx - 1)
                        prev_frame_idx = frame_idx - t_rel
                    else:
                        # the frame immediately after this frame (i.e. frame_idx + 1)
                        prev_frame_idx = frame_idx + t_rel
                else:
                    # for t_rel >= 2, we take the memory frame from every r-th frames
                    if not track_in_reverse:
                        # first find the nearest frame among every r-th frames before this frame
                        # for r=1, this would be (frame_idx - 2)
                        prev_frame_idx = ((frame_idx - 2) // stride) * stride
                        # then seek further among every r-th frames
                        prev_frame_idx = prev_frame_idx - (t_rel - 2) * stride
                    else:
                        # first find the nearest frame among every r-th frames after this frame
                        # for r=1, this would be (frame_idx + 2)
                        prev_frame_idx = -(-(frame_idx + 2) // stride) * stride
                        # then seek further among every r-th frames
                        prev_frame_idx = prev_frame_idx + (t_rel - 2) * stride
                out = output_dict["non_cond_frame_outputs"].get(prev_frame_idx, None)
                if out is None:
                    # If an unselected conditioning frame is among the last (self.num_maskmem - 1)
                    # frames, we still attend to it as if it's a non-conditioning frame.
                    out = unselected_cond_outputs.get(prev_frame_idx, None)
                prev_pose = torch.as_tensor(metas[prev_frame_idx]['pose'], device=self.device) if out is not None else None
                t_pos_and_prevs.append((t_pos, out, prev_pose, prev_frame_idx))

            for t_pos, prev, prev_pose, f_id in t_pos_and_prevs:
                if prev is None:
                    continue  # skip padding frames
                # "maskmem_features" might have been offloaded to CPU in demo use cases,
                # so we load it back to GPU (it's a no-op if it's already on GPU).
                feats = {k: v.to(device, non_blocking=True).flatten(2).permute(2, 0, 1) for k, v in prev["maskmem_features"].items()}
                to_cat_memory.append(feats)
                # Spatial positional encoding (it might have been offloaded to CPU in eval)
                tmp_feats = {k: x.permute(1, 2, 0).view(B, C, *feat_sizes[k]) if k == 'img' else x for k, x in feats.items()}
                if self.use_ego_pose_in_pe:
                    T_rel = lidar_aug_matrix @ torch.inverse(cur_pose) @ prev_pose @ torch.inverse(lidar_aug_matrix)
                else:
                    T_rel = None
                maskmem_enc, _, _ = self._positional_encoding(tmp_feats, metas[f_id], psam_infos[f_id], T_rel=T_rel)
                # if self.encode_ego_pose_in_mem:
                #     pose_enc = self.ego_pose_encoder.forward(cur_pose[None], prev_pose[None], lidar_aug_matrix).unsqueeze(
                #         0)  # (1, 1, mem_dim)
                #     maskmem_enc = {k: v + pose_enc for k, v in maskmem_enc.items()}
                # Temporal positional encoding
                maskmem_enc = {k: v + self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1] for k, v in maskmem_enc.items()}
                # Add+ tpos_sign_mul * maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1]
                to_cat_memory_pos_embed.append(maskmem_enc)

            # Construct the list of past object pointers
            if self.use_obj_ptrs_in_encoder:
                max_obj_ptrs_in_encoder = min(num_frames, self.max_obj_ptrs_in_encoder)
                # First add those object pointers from selected conditioning frames
                # (optionally, only include object pointers in the past during evaluation)
                if not self.training and self.only_obj_ptrs_in_the_past_for_eval:
                    ptr_cond_outputs = {
                        t: out
                        for t, out in selected_cond_outputs.items()
                        if (t >= frame_idx if track_in_reverse else t <= frame_idx)
                    }
                else:
                    ptr_cond_outputs = selected_cond_outputs
                pos_and_ptrs = [
                    # Temporal pos encoding contains how far away each pointer is from current frame
                    (
                        (
                            (frame_idx - t) * tpos_sign_mul
                            if self.use_signed_tpos_enc_to_obj_ptrs
                            else abs(frame_idx - t)
                        ),
                        out["obj_ptr"], t,
                    )
                    for t, out in ptr_cond_outputs.items()
                ]
                # Add up to (max_obj_ptrs_in_encoder - 1) non-conditioning frames before current frame
                for t_diff in range(1, max_obj_ptrs_in_encoder):
                    t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
                    if t < 0 or (num_frames is not None and t >= num_frames):
                        break
                    out = output_dict["non_cond_frame_outputs"].get(
                        t, unselected_cond_outputs.get(t, None)
                    )
                    if out is not None:
                        pos_and_ptrs.append((t_diff, out["obj_ptr"], t))
                # If we have at least one object pointer, add them to the across attention
                if len(pos_and_ptrs) > 0:
                    pos_list, ptrs_list, frame_list = zip(*pos_and_ptrs)
                    # stack object pointers along dim=0 into [ptr_seq_len, B, C] shape
                    obj_ptrs = {k: torch.stack([x[k] for x in ptrs_list], dim=0) for k in keys}
                    # a temporal positional embedding based on how far each object pointer is from
                    # the current frame (sine embedding normalized by the max pointer num).
                    if self.add_tpos_enc_to_obj_ptrs:
                        t_diff_max = max_obj_ptrs_in_encoder - 1
                        tpos_dim = C if self.proj_tpos_enc_in_obj_ptrs else self.mem_dim
                        obj_pos = torch.tensor(pos_list, device=device)
                        obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=tpos_dim)
                        obj_pos = self.pure_model.head.sam.obj_ptr_tpos_proj(obj_pos)
                        obj_pos = obj_pos.unsqueeze(1).expand(-1, B, self.mem_dim)
                        if self.encode_ego_pose_in_mem:
                            prev_poses = torch.stack([torch.as_tensor(metas[t]["pose"], device=self.device) for t in frame_list], dim=0)
                            pose_enc = self.ego_pose_encoder(cur_pose[None].expand_as(prev_poses), prev_poses, lidar_aug_matrix).unsqueeze(
                                1)
                            obj_pos = obj_pos + pose_enc
                    else:
                        obj_pos = obj_ptrs.new_zeros(len(pos_list), B, self.mem_dim)
                    if self.mem_dim < C:
                        # split a pointer into (C // self.mem_dim) tokens for self.mem_dim < C
                        obj_ptrs = {k: v.reshape(-1, B, C // self.mem_dim, self.mem_dim).permute(0, 2, 1, 3).flatten(0, 1) for k, v in
                                    obj_ptrs.items()}
                        obj_pos = obj_pos.repeat_interleave(C // self.mem_dim, dim=0)
                    obj_pos = {k: obj_pos for k in keys}
                    to_cat_memory.append(obj_ptrs)
                    to_cat_memory_pos_embed.append(obj_pos)
                    num_obj_ptr_tokens = obj_ptrs[keys[0]].shape[0]
                else:
                    num_obj_ptr_tokens = 0
        else:
            # for initial conditioning frames, encode them without using any previous memory
            if self.directly_add_no_mem_embed:
                # directly add no-mem embedding (instead of using the transformer encoder)
                pix_feat_with_mem = {}
                for k, v in current_vision_feat.items():
                    pix_feat_with_mem[k] = v + self.no_mem_embed
                    pix_feat_with_mem[k] = pix_feat_with_mem[k].permute(1, 2, 0).view(B, C, *feat_sizes[k])
                return pix_feat_with_mem

            # Use a dummy token on the first frame (to avoid empty memory input to tranformer encoder)
            to_cat_memory = [{k: self.no_mem_embed.expand(1, B, self.mem_dim) for k in keys}]
            to_cat_memory_pos_embed = [{k: self.no_mem_pos_enc.expand(1, B, self.mem_dim) for k in keys}]

        # Step 2: Concatenate the memories and forward through the transformer encoder
        # memory = {k: torch.cat([x[k] for x in to_cat_memory], dim=0) for k in keys}
        # memory_pos_embed = {k: torch.cat([x[k] for x in to_cat_memory_pos_embed], dim=0) for k in keys}
        # concat two modal together for multi-modal learning
        memory = torch.cat([torch.cat([x[k] for x in to_cat_memory]) for k in keys])
        memory = {k: memory for k in keys}
        memory_pos_embed = torch.cat([torch.cat([x[k] for x in to_cat_memory_pos_embed]) for k in keys])
        memory_pos_embed = {k: memory_pos_embed for k in keys}
        # add pts current feat to image memory! vice versa
        if 'img' in self.modal and 'pts' in self.modal:
            memory['img'] = torch.cat([memory['img'], current_vision_feat['pts']])
            memory_pos_embed['img'] = torch.cat([memory_pos_embed['img'], current_vision_pos_embed['pts']])
            memory['pts'] = torch.cat([memory['pts'], current_vision_feat['img']])
            memory_pos_embed['pts'] = torch.cat([memory_pos_embed['pts'], current_vision_pos_embed['img']])

        pix_feat_with_mem = self.pure_model.fusion.memory_attention(  # MemoryAttention
            curr=current_vision_feat,
            curr_pos=current_vision_pos_embed,
            memory=memory,
            memory_pos=memory_pos_embed,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        # reshape the output (HW)BC => BCHW
        pix_feat_with_mem = {k: v.permute(1, 2, 0).view(B, C, *feat_sizes[k]) for k, v in pix_feat_with_mem.items()}
        return pix_feat_with_mem

    def _encode_new_memory(
            self,
            current_vision_feats,
            feat_sizes,
            pred_masks_high_res,
            object_score_logits,
            is_mask_from_pts,
            psam_info,
    ):
        """Encode the current image and its prediction into a memory feature."""
        keys = list(current_vision_feats.keys())
        B = current_vision_feats[keys[0]][-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        ret_maskmem_features, ret_maskmem_pos_enc = {}, {}
        for key in keys:
            if key == 'pts':
                # top-level feature, (HW)BC => B(HW)C
                pix_feat = current_vision_feats[key][-1].permute(1, 0, 2)
            else:  # img
                # top-level feature, (HW)BC => BCHW
                pix_feat = current_vision_feats[key][-1].permute(1, 2, 0).view(B, C, *feat_sizes[key][-1])
            if self.non_overlap_masks_for_mem_enc and not self.training:
                # optionally, apply non-overlapping constraints to the masks (it's applied
                # in the batch dimension and should only be used during eval, where all
                # the objects come from the same video under batch size 1).
                pred_masks_high_res[key] = self._apply_non_overlapping_constraints(pred_masks_high_res[key])
            # scale the raw mask logits with a temperature before applying sigmoid
            binarize = self.binarize_mask_from_pts_for_mem_enc and is_mask_from_pts
            if binarize and not self.training:
                mask_for_mem = (pred_masks_high_res[key] > 0).float()
            else:
                # apply sigmoid on the raw mask logits to turn them into range (0, 1)
                mask_for_mem = torch.sigmoid(pred_masks_high_res[key])
            # apply scale and bias terms to the sigmoid probabilities
            if self.sigmoid_scale_for_mem_enc != 1.0:
                mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
            if self.sigmoid_bias_for_mem_enc != 0.0:
                mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc
            if key == 'img':
                maskmem_out = self.pure_model.fusion.memory_encoder.image(  # MemoryEncoder
                    pix_feat, mask_for_mem, skip_mask_sigmoid=True  # sigmoid already applied
                )
            else:
                maskmem_out = self.pure_model.fusion.memory_encoder.point(  # PtsMemoryEncoder
                    pix_feat, mask_for_mem, psam_info=psam_info, skip_mask_sigmoid=True  # sigmoid already applied
                )
            maskmem_features = maskmem_out["vision_features"]
            # add a no-object embedding to the spatial memory to indicate that the frame
            # is predicted to be occluded (i.e. no object is appearing in the frame)
            if self.no_obj_embed_spatial is not None:
                is_obj_appearing = (object_score_logits[key] > 0).float()
                maskmem_features += (1 - is_obj_appearing[..., None, None]) * self.no_obj_embed_spatial[..., None, None].expand(
                    *maskmem_features.shape)
            ret_maskmem_features[key] = maskmem_features

        return ret_maskmem_features

    def _positional_encoding(self, feats, meta, psam_info=None, sam_info=None, T_rel=None):
        # feats['img']: BCHW, feats['pts']: (n_pts, B, C)
        is_multi_modal = 'img' in feats and 'pts' in feats
        pos_embeds = {}
        if 'img' in feats:
            pos_embeds['img'] = self.union_pe_func.get_image_dense_pe(feats['img']).to(feats['img'].dtype)  # BCHW
            if is_multi_modal:
                pos_embeds['img'] = pos_embeds['img'] + self.union_pe_func.get_union_image_pe(meta, T_rel=T_rel)
            if sam_info is not None:
                sam_info['vision_pos_enc'] = pos_embeds['img'][0]  # used in sam head CHW
            pos_embeds['img'] = pos_embeds['img'].view(feats['img'].size(0), feats['img'].size(1), -1).permute(2, 0, 1)

        if 'pts' in feats:
            assert psam_info is not None
            pts_pos = psam_info['pts_pos']
            if T_rel is not None:
                T_rel = T_rel.to(pts_pos.device).to(pts_pos.dtype)
                pts_pos_hom = torch.cat([pts_pos, pts_pos.new_ones(pts_pos.shape[0], 1)], dim=-1)  # Nx4
                pts_pos = pts_pos_hom @ T_rel.T[:, :3]
            pos_embeds['pts'] = self.union_pe_func.get_point_pe(pts_pos).to(feats['pts'].dtype)
            if is_multi_modal:
                pos_embeds['pts'] = pos_embeds['pts'] + self.union_pe_func.get_union_point_pe(pts_pos)
            psam_info['pts_pos_enc'] = pos_embeds['pts']  # used in sam head
            pos_embeds['pts'] = pos_embeds['pts'].unsqueeze(1).expand_as(feats['pts'])

        return pos_embeds, sam_info, psam_info

    def _track_step(
            self,
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            feat_sizes,
            point_inputs,
            mask_inputs,
            output_dict,
            num_frames,
            track_in_reverse,
            prev_sam_mask_logits,
            psam_infos=None,
            metas=None,
    ):
        assert isinstance(current_vision_feats, dict), f'got {type(current_vision_feats)}'
        assert isinstance(feat_sizes, dict), f'got {type(feat_sizes)}'
        current_out = {"point_inputs": point_inputs, "mask_inputs": mask_inputs}
        # High-resolution feature maps for the SAM head, reshape (HW)BC => BCHW
        high_res_features = {}
        if self.use_high_res_features_in_sam:
            if 'img' in current_vision_feats:
                assert len(current_vision_feats['img']) > 1, f'got {len(current_vision_feats["img"])}'
                high_res_features['img'] = [
                    x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                    for x, s in zip(current_vision_feats['img'][:-1], feat_sizes['img'][:-1])
                ]
            if 'pts' in current_vision_feats:
                high_res_features['pts'] = current_vision_feats['pts'][:-1]

        cur_feats = {
            k: x[-1].permute(1, 2, 0).view(x[-1].size(1), x[-1].size(2), *feat_sizes[k][-1]) if k == 'img' else x[-1]
            for k, x in current_vision_feats.items()
        }

        current_vision_pos_embeds, sam_info, psam_info \
            = self._positional_encoding(cur_feats, metas[frame_idx], psam_info=psam_infos[frame_idx], sam_info={})

        if mask_inputs is not None and self.use_mask_input_as_output_without_sam:
            # When use_mask_input_as_output_without_sam=True, we directly output the mask input
            # (see it as a GT mask) without using a SAM prompt encoder + mask decoder.
            pix_feat = {}
            for key in current_vision_feats:
                pix_feat[key] = current_vision_feats[key][-1].permute(1, 2, 0)
                pix_feat[key] = pix_feat[key].view(-1, self.hidden_dim, *feat_sizes[key][-1])
            sam_outputs = self._use_mask_as_output(
                pix_feat, high_res_features, mask_inputs, psam_info=psam_info, sam_info=sam_info, meta=metas[frame_idx],
            )
        else:
            # fused the visual feature with previous memory features in the memory bank
            pix_feat = self._prepare_memory_conditioned_features(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feat={k: v[-1] for k, v in current_vision_feats.items()},
                current_vision_pos_embed=current_vision_pos_embeds,
                feat_sizes={k: v[-1] for k, v in feat_sizes.items()},
                output_dict=output_dict,
                num_frames=num_frames,
                track_in_reverse=track_in_reverse,
                metas=metas,
                psam_infos=psam_infos,
            )
            # apply SAM-style segmentation head
            # here we might feed previously predicted low-res SAM mask logits into the SAM mask decoder,
            # e.g. in demo where such logits come from earlier interaction instead of correction sampling
            # (in this case, any `mask_inputs` shouldn't reach here as they are sent to _use_mask_as_output instead)
            if prev_sam_mask_logits is not None:
                assert point_inputs is not None and mask_inputs is None
                mask_inputs = prev_sam_mask_logits
            multimask_output = self._use_multimask(is_init_cond_frame, point_inputs)
            sam_outputs = self.pure_model.head.sam(
                backbone_features=pix_feat,
                point_inputs=point_inputs,
                mask_inputs=mask_inputs,
                high_res_features=high_res_features,
                multimask_output=multimask_output,
                psam_info=psam_info,
                union_pe_func=self.union_pe_func,
                sam_info=sam_info,
                meta=metas[frame_idx],
            )

        return current_out, sam_outputs, high_res_features, pix_feat, sam_info, psam_info

    def _encode_memory_in_output(
            self,
            current_vision_feats,
            feat_sizes,
            point_inputs,
            run_mem_encoder,
            high_res_masks,
            object_score_logits,
            current_out,
            psam_info,
    ):
        if run_mem_encoder and self.num_maskmem > 0:
            high_res_masks_for_mem_enc = high_res_masks
            maskmem_features = self._encode_new_memory(
                current_vision_feats=current_vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_masks_for_mem_enc,
                object_score_logits=object_score_logits,
                is_mask_from_pts=(point_inputs is not None),
                psam_info=psam_info,
            )
            current_out["maskmem_features"] = maskmem_features
        else:
            current_out["maskmem_features"] = None

    def track_step(
            self,
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            feat_sizes,
            point_inputs,
            mask_inputs,
            output_dict,
            num_frames,
            track_in_reverse=False,  # tracking in reverse time order (for demo usage)
            # Whether to run the memory encoder on the predicted masks. Sometimes we might want
            # to skip the memory encoder with `run_mem_encoder=False`. For example,
            # in demo we might call `track_step` multiple times for each user click,
            # and only encode the memory when the user finalizes their clicks. And in ablation
            # settings like SAM training on static images, we don't need the memory encoder.
            run_mem_encoder=True,
            # The previously predicted SAM mask logits (which can be fed together with new clicks in demo).
            prev_sam_mask_logits=None,
            psam_infos=None,
            metas=None,
    ):
        current_out, sam_outputs, _, _, _, _ = self._track_step(
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            feat_sizes,
            point_inputs,
            mask_inputs,
            output_dict,
            num_frames,
            track_in_reverse,
            prev_sam_mask_logits,
            psam_infos,
            metas,
        )

        low_res_masks = {k: v['low_res_masks'] for k, v in sam_outputs.items()}
        high_res_masks = {k: v['high_res_masks'] for k, v in sam_outputs.items()}
        obj_ptr = {k: v['obj_ptr'] for k, v in sam_outputs.items()}
        object_score_logits = {k: v['object_score_logits'] for k, v in sam_outputs.items()}

        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr
        if not self.training:
            # Only add this in inference (to avoid unused param in activation checkpointing;
            # it's mainly used in the demo to encode spatial memories w/ consolidated masks)
            current_out["object_score_logits"] = object_score_logits

        # Finally run the memory encoder on the predicted mask to encode
        # it into a new memory feature (that can be used in future frames)
        self._encode_memory_in_output(
            current_vision_feats,
            feat_sizes,
            point_inputs,
            run_mem_encoder,
            high_res_masks,
            object_score_logits,
            current_out,
            psam_infos[frame_idx],
        )

        return current_out

    def _use_multimask(self, is_init_cond_frame, point_inputs):
        """Whether to use multimask output in the SAM head."""
        num_pts = 0 if point_inputs is None else list(point_inputs.values())[0]["point_labels"].size(1)
        multimask_output = (
                self.multimask_output_in_sam
                and (is_init_cond_frame or self.multimask_output_for_tracking)
                and (self.multimask_min_pt_num <= num_pts <= self.multimask_max_pt_num)
        )
        return multimask_output

    def _apply_non_overlapping_constraints(self, pred_masks):
        """
        Apply non-overlapping constraints to the object scores in pred_masks. Here we
        keep only the highest scoring object at each spatial location in pred_masks.
        """
        batch_size = pred_masks.size(0)
        if batch_size == 1:
            return pred_masks

        device = pred_masks.device
        # "max_obj_inds": object index of the object with the highest score at each location
        max_obj_inds = torch.argmax(pred_masks, dim=0, keepdim=True)
        # "batch_obj_inds": object index of each object slice (along dim 0) in `pred_masks`
        batch_obj_inds = torch.arange(batch_size, device=device)[:, None, None, None]
        keep = max_obj_inds == batch_obj_inds
        # suppress overlapping regions' scores below -10.0 so that the foreground regions
        # don't overlap (here sigmoid(-10.0)=4.5398e-05)
        pred_masks = torch.where(keep, pred_masks, torch.clamp(pred_masks, max=-10.0))
        return pred_masks
