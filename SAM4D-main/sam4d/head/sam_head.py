from typing import List, Optional, Tuple, Type, Dict

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
import torchsparse.nn as spnn

from sam4d.utils import LayerNorm2d, MLP, TwoWayTransformer, PromptEncoder, PromptEncoderPC
from sam4d.sam4d_base import NO_OBJ_SCORE
from sam4d.pv_cnn_utils import BasicDeconvolutionBlock, voxel_to_point
from torchsparse.tensor import SparseTensor, PointTensor
from mmengine.registry import MODELS


@MODELS.register_module()
class SAMHead(nn.Module):
    def __init__(self,
                 transformer: dict,
                 modal=dict(img=dict(image_size=1024, backbone_stride=16)),
                 num_multimask_outputs=3,
                 transformer_dim=256,
                 mem_dim=256,
                 iou_head_depth=3,
                 iou_head_hidden_dim=256,
                 use_high_res_features=False,
                 iou_prediction_use_sigmoid=False,
                 pred_obj_scores=False,
                 pred_obj_scores_mlp=False,
                 use_multimask_token_for_obj_ptr=False,
                 use_obj_ptrs_in_encoder=False,
                 proj_tpos_enc_in_obj_ptrs=False,
                 use_mlp_for_obj_ptr_proj=False,
                 # Soft no object, i.e. mix in no_obj_ptr softly,
                 # hope to make recovery easier if there is a mistake and mitigate accumulation of errors
                 soft_no_obj_ptr: bool = False,
                 fixed_no_obj_ptr: bool = False,
                 sam_mask_decoder_extra_args={},
                 init_cfg=None,
                 **kwargs):
        super(SAMHead, self).__init__()
        self.sam_prompt_embed_dim = transformer_dim
        self.hidden_dim = self.sam_prompt_embed_dim
        self.mem_dim = mem_dim
        self.pred_obj_scores = pred_obj_scores
        self.soft_no_obj_ptr = soft_no_obj_ptr
        self.fixed_no_obj_ptr = fixed_no_obj_ptr

        self.modal = modal
        if 'img' in modal:
            self.image_size = modal['img']['image_size']
            self.sam_image_embedding_size = self.image_size // modal['img']['backbone_stride']

            # build PromptEncoder and MaskDecoder from SAM
            # (their hyperparameters like `mask_in_chans=16` are from SAM code)
            self.sam_prompt_encoder = PromptEncoder(
                embed_dim=self.sam_prompt_embed_dim,
                image_embedding_size=(
                    self.sam_image_embedding_size,
                    self.sam_image_embedding_size,
                ),
                input_image_size=(self.image_size, self.image_size),
                mask_in_chans=16,
            )
        if 'pts' in modal:
            # prompt embedding from PSAM head
            self.psam_prompt_encoder = PromptEncoderPC(transformer_dim, mask_in_chans=16)

        self.sam_mask_decoder = MaskDecoder(
            modal,
            num_multimask_outputs=num_multimask_outputs,
            transformer=transformer,
            transformer_dim=self.sam_prompt_embed_dim,
            iou_head_depth=iou_head_depth,
            iou_head_hidden_dim=iou_head_hidden_dim,
            use_high_res_features=use_high_res_features,
            iou_prediction_use_sigmoid=iou_prediction_use_sigmoid,
            pred_obj_scores=pred_obj_scores,
            pred_obj_scores_mlp=pred_obj_scores_mlp,
            use_multimask_token_for_obj_ptr=use_multimask_token_for_obj_ptr,
            **sam_mask_decoder_extra_args,
        )
        if use_obj_ptrs_in_encoder:
            # a linear projection on SAM output tokens to turn them into object pointers
            self.obj_ptr_proj = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
            if use_mlp_for_obj_ptr_proj:
                self.obj_ptr_proj = MLP(
                    self.hidden_dim, self.hidden_dim, self.hidden_dim, 3
                )
        else:
            self.obj_ptr_proj = torch.nn.Identity()
        if proj_tpos_enc_in_obj_ptrs:
            # a linear projection on temporal positional encoding in object pointers to
            # avoid potential interference with spatial positional encoding
            self.obj_ptr_tpos_proj = torch.nn.Linear(self.hidden_dim, self.mem_dim)
        else:
            self.obj_ptr_tpos_proj = torch.nn.Identity()

        if self.pred_obj_scores and use_obj_ptrs_in_encoder:
            self.no_obj_ptr = torch.nn.Parameter(torch.zeros(1, self.hidden_dim))
            trunc_normal_(self.no_obj_ptr, std=0.02)

    def forward(
            self,
            backbone_features,
            point_inputs=None,
            mask_inputs=None,
            high_res_features=None,
            multimask_output=False,
            psam_info=None,
            union_pe_func=None,
            sam_info=None,
            meta=None,
    ):
        """
        Forward SAM prompt encoders and mask heads.

        Inputs:
        - backbone_features: image features of [B, C, H, W] shape
        - point_inputs: a dictionary with "point_coords" and "point_labels", where
          1) "point_coords" has [B, P, 2] shape and float32 dtype and contains the
             absolute pixel-unit coordinate in (x, y) format of the P input points
          2) "point_labels" has shape [B, P] and int32 dtype, where 1 means
             positive clicks, 0 means negative clicks, and -1 means padding
        - mask_inputs: a mask of [B, 1, H*16, W*16] shape, float or bool, with the
          same spatial size as the image.
        - high_res_features: either 1) None or 2) or a list of length 2 containing
          two feature maps of [B, C, 4*H, 4*W] and [B, C, 2*H, 2*W] shapes respectively,
          which will be used as high-resolution feature maps for SAM decoder.
        - multimask_output: if it's True, we output 3 candidate masks and their 3
          corresponding IoU estimates, and if it's False, we output only 1 mask and
          its corresponding IoU estimate.

        Outputs:
        - low_res_multimasks: [B, M, H*4, W*4] shape (where M = 3 if
          `multimask_output=True` and M = 1 if `multimask_output=False`), the SAM
          output mask logits (before sigmoid) for the low-resolution masks, with 4x
          the resolution (1/4 stride) of the input backbone_features.
        - high_res_multimasks: [B, M, H*16, W*16] shape (where M = 3
          if `multimask_output=True` and M = 1 if `multimask_output=False`),
          upsampled from the low-resolution masks, with shape size as the image
          (stride is 1 pixel).
        - ious, [B, M] shape, where (where M = 3 if `multimask_output=True` and M = 1
          if `multimask_output=False`), the estimated IoU of each output mask.
        - low_res_masks: [B, 1, H*4, W*4] shape, the best mask in `low_res_multimasks`.
          If `multimask_output=True`, it's the mask with the highest IoU estimate.
          If `multimask_output=False`, it's the same as `low_res_multimasks`.
        - high_res_masks: [B, 1, H*16, W*16] shape, the best mask in `high_res_multimasks`.
          If `multimask_output=True`, it's the mask with the highest IoU estimate.
          If `multimask_output=False`, it's the same as `high_res_multimasks`.
        - obj_ptr: [B, C] shape, the object pointer vector for the output mask, extracted
          based on the output token from the SAM mask decoder.
        """
        assert isinstance(backbone_features, dict), f'backbone_features should be a dict, but got {type(backbone_features)}'
        keys = list(backbone_features.keys())
        B = backbone_features[keys[0]].size(0)
        device = backbone_features[keys[0]].device
        for key in keys:
            assert key in ['img', 'pts'], f'key should be in ["img", "pts"], but got {key}'
            assert backbone_features[key].size(1) == self.sam_prompt_embed_dim
            if key == 'img':
                assert backbone_features[key].size(2) == self.sam_image_embedding_size
                assert backbone_features[key].size(3) == self.sam_image_embedding_size

        # a) Handle point prompts
        sam_point_prompt = {}
        point_inputs = {} if point_inputs is None else point_inputs
        assert isinstance(point_inputs, dict), f'point_inputs should be a dict, but got {type(point_inputs)}'
        for key in keys:
            if key in point_inputs:
                sam_point_coords = point_inputs[key]["point_coords"]
                sam_point_labels = point_inputs[key]["point_labels"]
                assert sam_point_coords.size(0) == B and sam_point_labels.size(0) == B
                sam_point_prompt[key] = (sam_point_coords, sam_point_labels)
            else:
                # If no points are provide, pad with an empty point (with label -1)
                sam_point_coords = torch.zeros(B, 1, 2 if key == 'img' else 3, device=device)
                sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=device)
                sam_point_prompt[key] = (sam_point_coords, sam_point_labels)

        # b) Handle mask prompts
        if mask_inputs is not None:
            assert isinstance(mask_inputs, dict), f'mask_inputs should be a dict, but got {type(mask_inputs)}'
            # If mask_inputs is provided, downsize it into low-res mask input if needed
            # and feed it as a dense mask prompt into the SAM mask encoder
            sam_mask_prompt = {}
            for key in keys:
                tmp_mask_inputs = mask_inputs[key]
                assert tmp_mask_inputs.ndim == 4 and tmp_mask_inputs.shape[:2] == (B, 1)
                if key == 'img' and tmp_mask_inputs.shape[-2:] != self.sam_prompt_encoder.mask_input_size:
                    sam_mask_prompt[key] = F.interpolate(
                        tmp_mask_inputs.float(),
                        size=self.sam_prompt_encoder.mask_input_size,
                        align_corners=False,
                        mode="bilinear",
                        antialias=True,  # use antialias for downsampling
                    )
                else:
                    sam_mask_prompt[key] = tmp_mask_inputs
        else:
            # Otherwise, simply feed None (and SAM's prompt encoder will add
            # a learned `no_mask_embed` to indicate no mask input in this case).
            sam_mask_prompt = {k: None for k in keys}

        sparse_embeddings, dense_embeddings, image_pe = {}, {}, {}
        for key in keys:
            if key == 'img':
                sparse_embeddings[key], dense_embeddings[key] = self.sam_prompt_encoder.forward(
                    union_pe_func=union_pe_func,
                    points=sam_point_prompt[key],
                    boxes=None,
                    masks=sam_mask_prompt[key],
                    multi_modal='img' in keys and 'pts' in keys,
                    meta=meta,
                )
                # image_pe[key] = self.sam_prompt_encoder.get_dense_pe()
                image_pe[key] = sam_info['vision_pos_enc'].unsqueeze(0)
            else:
                assert key == 'pts', f'got key {key}'
                sparse_embeddings[key], dense_embeddings[key] = self.psam_prompt_encoder.forward(
                    union_pe_func=union_pe_func,
                    points=sam_point_prompt[key],
                    boxes=None,
                    masks=sam_mask_prompt[key],
                    psam_info=psam_info,
                    sub_pts_num=backbone_features[key].shape[2],  # in (B, C, n_pts, 1) format
                    multi_modal='img' in keys and 'pts' in keys,
                )
                image_pe[key] = psam_info['pts_pos_enc']

        # add one modal point prompt to the other
        if 'img' in keys and 'pts' in keys:
            sparse_embeddings['pts'] = torch.cat([sparse_embeddings['pts'], sparse_embeddings['img']], dim=1)
            sparse_embeddings['img'] = torch.cat([sparse_embeddings['img'], sparse_embeddings['pts']], dim=1)

        sam_outs = self.sam_mask_decoder.forward(
            image_embeddings=backbone_features,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=False,  # the image is already batched
            high_res_features=high_res_features,
            psam_info=psam_info,
        )

        for key, sam_out in sam_outs.items():
            low_res_multimasks = sam_out.pop('masks')
            ious = sam_out.pop('iou_pred')
            sam_output_tokens = sam_out.pop('sam_tokens_out')
            object_score_logits = sam_out.pop('object_score_logits')

            if self.pred_obj_scores:
                is_obj_appearing = object_score_logits > 0

                # Mask used for spatial memories is always a *hard* choice between obj and no obj,
                # consistent with the actual mask prediction
                low_res_multimasks = torch.where(
                    is_obj_appearing[:, None, None],
                    low_res_multimasks,
                    NO_OBJ_SCORE,
                )

            # convert masks from possibly bfloat16 (or float16) to float32
            # (older PyTorch versions before 2.1 don't support `interpolate` on bf16)
            low_res_multimasks = low_res_multimasks.float()
            if key == 'img':
                high_res_multimasks = F.interpolate(
                    low_res_multimasks,
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                assert key == 'pts', f'got key {key}'
                high_res_multimasks = []
                sp_info = psam_info['pts_sp_tensor_info'][0]
                for one_low_res_mask in low_res_multimasks:
                    src_i = SparseTensor(one_low_res_mask.squeeze(2).transpose(0, 1), sp_info['coords'], sp_info['stride'],
                                         spatial_range=sp_info['spatial_range']).set_caches(sp_info['_cache'])
                    src_i = voxel_to_point(src_i, psam_info['pts_org_feats'])
                    high_res_multimasks.append(src_i.F)
                high_res_multimasks = torch.stack(high_res_multimasks, dim=0).permute(0, 2, 1).unsqueeze(3)

            sam_output_token = sam_output_tokens[:, 0]
            if multimask_output:
                # take the best mask prediction (with the highest IoU estimation)
                best_iou_inds = torch.argmax(ious, dim=-1)
                batch_inds = torch.arange(B, device=device)
                low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
                high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
                if sam_output_tokens.size(1) > 1:
                    sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
            else:
                low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks

            # Extract object pointer from the SAM output token (with occlusion handling)
            obj_ptr = self.obj_ptr_proj(sam_output_token)
            if self.pred_obj_scores:
                # Allow *soft* no obj ptr, unlike for masks
                if self.soft_no_obj_ptr:
                    lambda_is_obj_appearing = object_score_logits.sigmoid()
                else:
                    lambda_is_obj_appearing = is_obj_appearing.float()

                if self.fixed_no_obj_ptr:
                    obj_ptr = lambda_is_obj_appearing * obj_ptr
                obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

            sam_out['low_res_multimasks'] = low_res_multimasks
            sam_out['high_res_multimasks'] = high_res_multimasks
            sam_out['ious'] = ious
            sam_out['low_res_masks'] = low_res_masks
            sam_out['high_res_masks'] = high_res_masks
            sam_out['obj_ptr'] = obj_ptr
            sam_out['object_score_logits'] = object_score_logits
            sam_outs[key] = sam_out
        return sam_outs


class MaskDecoder(nn.Module):
    def __init__(
            self,
            modal,
            transformer_dim: int,
            transformer: dict,
            num_multimask_outputs: int = 3,
            activation: Type[nn.Module] = nn.GELU,
            iou_head_depth: int = 3,
            iou_head_hidden_dim: int = 256,
            use_high_res_features: bool = False,
            iou_prediction_use_sigmoid=False,
            dynamic_multimask_via_stability=False,
            dynamic_multimask_stability_delta=0.05,
            dynamic_multimask_stability_thresh=0.98,
            pred_obj_scores: bool = False,
            pred_obj_scores_mlp: bool = False,
            use_multimask_token_for_obj_ptr: bool = False,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = TwoWayTransformer(**transformer)

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.pred_obj_scores = pred_obj_scores
        if self.pred_obj_scores:
            self.obj_score_token = nn.Embedding(1, transformer_dim)
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr

        if 'img' in modal:
            self.output_upscaling = nn.Sequential(
                nn.ConvTranspose2d(
                    transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
                ),
                LayerNorm2d(transformer_dim // 4),
                activation(),
                nn.ConvTranspose2d(
                    transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
                ),
                activation(),
            )
        if 'pts' in modal:
            no_act = True
            act = spnn.LeakyReLU(1, True) if no_act else spnn.LeakyReLU(0.1, True)
            self.output_upscaling_pts = nn.Sequential(
                spnn.Conv3d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2, transposed=True),  # , generative=True),
                spnn.BatchNorm(transformer_dim // 4),
                act,
                spnn.Conv3d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2, transposed=True, bias=True),
                # generative=True),
                act,
            )

        self.use_high_res_features = use_high_res_features
        if use_high_res_features and 'img' in modal:
            self.conv_s0 = nn.Conv2d(
                transformer_dim, transformer_dim // 8, kernel_size=1, stride=1
            )
            self.conv_s1 = nn.Conv2d(
                transformer_dim, transformer_dim // 4, kernel_size=1, stride=1
            )
        if use_high_res_features and 'pts' in modal:
            self.conv_s0_pts = spnn.Conv3d(
                transformer_dim // 2, transformer_dim // 8, kernel_size=1, stride=1
            )
            self.conv_s1_pts = spnn.Conv3d(
                transformer_dim, transformer_dim // 4, kernel_size=1, stride=1
            )

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim,
            iou_head_hidden_dim,
            self.num_mask_tokens,
            iou_head_depth,
            sigmoid_output=iou_prediction_use_sigmoid,
        )
        if self.pred_obj_scores:
            self.pred_obj_score_head = nn.Linear(transformer_dim, 1)
            if pred_obj_scores_mlp:
                self.pred_obj_score_head = MLP(transformer_dim, transformer_dim, 1, 3)

        # When outputting a single mask, optionally we can dynamically fall back to the best
        # multimask output token if the single mask output token gives low stability scores.
        self.dynamic_multimask_via_stability = dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta = dynamic_multimask_stability_delta
        self.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh

    def forward(
            self,
            image_embeddings: Dict[str, torch.Tensor],
            image_pe: Dict[str, torch.Tensor],
            sparse_prompt_embeddings: Dict[str, torch.Tensor],
            dense_prompt_embeddings: Dict[str, torch.Tensor],
            multimask_output: bool,
            repeat_image: bool,
            high_res_features: Optional[Dict[str, List[torch.Tensor]]] = None,
            psam_info=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
          torch.Tensor: batched SAM token for mask output
        """
        sam_outs = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            repeat_image=repeat_image,
            high_res_features=high_res_features,
            psam_info=psam_info,
        )

        for key, sam_out in sam_outs.items():
            masks = sam_out.pop('masks')
            iou_pred = sam_out.pop('iou_pred')
            mask_tokens_out = sam_out.pop('mask_tokens_out')

            # Select the correct mask or masks for output
            if multimask_output:
                masks = masks[:, 1:, :, :]
                iou_pred = iou_pred[:, 1:]
            elif self.dynamic_multimask_via_stability and not self.training:
                masks, iou_pred = self._dynamic_multimask_via_stability(masks, iou_pred)
            else:
                masks = masks[:, 0:1, :, :]
                iou_pred = iou_pred[:, 0:1]

            if multimask_output and self.use_multimask_token_for_obj_ptr:
                sam_tokens_out = mask_tokens_out[:, 1:]  # [b, 3, c] shape
            else:
                # Take the mask output token. Here we *always* use the token for single mask output.
                # At test time, even if we track after 1-click (and using multimask_output=True),
                # we still take the single mask token here. The rationale is that we always track
                # after multiple clicks during training, so the past tokens seen during training
                # are always the single mask token (and we'll let it be the object-memory token).
                sam_tokens_out = mask_tokens_out[:, 0:1]  # [b, 1, c] shape
            sam_out['masks'] = masks
            sam_out['iou_pred'] = iou_pred
            sam_out['sam_tokens_out'] = sam_tokens_out
            sam_outs[key] = sam_out

        # Prepare output
        return sam_outs

    def predict_masks(
            self,
            image_embeddings: Dict[str, torch.Tensor],
            image_pe: Dict[str, torch.Tensor],
            sparse_prompt_embeddings: Dict[str, torch.Tensor],
            dense_prompt_embeddings: Dict[str, torch.Tensor],
            repeat_image: bool,
            high_res_features: Optional[Dict[str, List[torch.Tensor]]] = None,
            psam_info=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        s = 0
        if self.pred_obj_scores:
            output_tokens = torch.cat(
                [
                    self.obj_score_token.weight,
                    self.iou_token.weight,
                    self.mask_tokens.weight,
                ],
                dim=0,
            )
            s = 1
        else:
            output_tokens = torch.cat(
                [self.iou_token.weight, self.mask_tokens.weight], dim=0
            )
        sam_outs = {}
        keys = list(image_embeddings.keys())
        for key in keys:
            if key == 'img':
                sam_outs[key] = self.predict_img_masks(
                    s,
                    output_tokens,
                    image_embeddings[key],
                    image_pe[key],
                    sparse_prompt_embeddings[key],
                    dense_prompt_embeddings[key],
                    repeat_image,
                    high_res_features[key] if self.use_high_res_features else None,
                )
            else:
                assert key == 'pts'
                sam_outs[key] = self.predict_pts_masks(
                    s,
                    output_tokens,
                    image_embeddings[key],
                    image_pe[key],
                    sparse_prompt_embeddings[key],
                    dense_prompt_embeddings[key],
                    repeat_image,
                    high_res_features[key] if self.use_high_res_features else None,
                    psam_info=psam_info,
                )
        return sam_outs

    def predict_img_masks(
            self,
            s: int,
            output_tokens: torch.Tensor,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
            repeat_image: bool,
            high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        if repeat_image:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            assert image_embeddings.shape[0] == tokens.shape[0]
            src = image_embeddings
        src = src + dense_prompt_embeddings
        assert (image_pe.size(0) == 1), "image_pe should have size 1 in batch dim (from `get_dense_pe()`)"
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        # # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        src = src.flatten(2).permute(0, 2, 1)
        pos_src = pos_src.flatten(2).permute(0, 2, 1)
        hs, src = self.transformer.forward(src, pos_src, tokens)
        iou_token_out = hs[:, s, :]
        mask_tokens_out = hs[:, s + 1: (s + 1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        if not self.use_high_res_features:
            upscaled_embedding = self.output_upscaling(src)
        else:
            dc1, ln1, act1, dc2, act2 = self.output_upscaling
            feat_s0, feat_s1 = high_res_features
            upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)
        if self.pred_obj_scores:
            assert s == 1
            object_score_logits = self.pred_obj_score_head(hs[:, 0, :])
        else:
            # Obj scores logits - default to 10.0, i.e. assuming the object is present, sigmoid(10)=1
            object_score_logits = 10.0 * iou_pred.new_ones(iou_pred.shape[0], 1)

        ret = dict(
            masks=masks, iou_pred=iou_pred, mask_tokens_out=mask_tokens_out,
            object_score_logits=object_score_logits
        )
        return ret

    def predict_pts_masks(
            self,
            s: int,
            output_tokens: torch.Tensor,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
            repeat_image: bool,
            high_res_features: Optional[List[torch.Tensor]] = None,
            psam_info: Dict = None,
    ) -> Dict[str, torch.Tensor]:
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        assert image_embeddings.ndim == 4 and image_embeddings.shape[-1] == 1, f'got {image_embeddings.shape}'
        image_embeddings = image_embeddings.squeeze(3).transpose(1, 2)  # B, n_pts, C
        if repeat_image:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            assert image_embeddings.shape[0] == tokens.shape[0]
            src = image_embeddings
        assert src.shape == dense_prompt_embeddings.shape, f'got {src.shape}, {dense_prompt_embeddings.shape}'
        src = src + dense_prompt_embeddings
        assert image_pe.ndim == 2, f'got {image_pe.shape}'
        pos_src = torch.repeat_interleave(image_pe.unsqueeze(0), tokens.shape[0], dim=0)

        # Run the transformer
        hs, src = self.transformer.forward(src, pos_src, tokens)
        iou_token_out = hs[:, s, :]
        mask_tokens_out = hs[:, s + 1: (s + 1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        sp_info = psam_info['pts_sp_tensor_info'][-1]
        upscaled_embedding = []
        feat_s0, feat_s1 = high_res_features
        for i in range(src.shape[0]):
            src_i = SparseTensor(src[i], sp_info['coords'], sp_info['stride'], sp_info['spatial_range']).set_caches(sp_info['_cache'])
            if not self.use_high_res_features:
                src_i = self.output_upscaling_pts(src_i)
            else:
                dc1, ln1, act1, dc2, act2 = self.output_upscaling_pts
                src_i = dc1(src_i)
                src_i.F += feat_s1[:, i, :]
                src_i = act1(ln1(src_i))
                src_i = dc2(src_i)
                src_i.F += feat_s0[:, i, :]
                src_i = act2(src_i)
            # src_i = voxel_to_point(src_i, psam_info['pts_org_feats'])
            upscaled_embedding.append(src_i.F)
        upscaled_embedding = torch.stack(upscaled_embedding, dim=0)

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)
        masks = (hyper_in @ upscaled_embedding.permute(0, 2, 1)).unsqueeze(3)  # (B, n_mask, n_pts, 1)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)
        if self.pred_obj_scores:
            assert s == 1
            object_score_logits = self.pred_obj_score_head(hs[:, 0, :])
        else:
            # Obj scores logits - default to 10.0, i.e. assuming the object is present, sigmoid(10)=1
            object_score_logits = 10.0 * iou_pred.new_ones(iou_pred.shape[0], 1)

        ret = dict(
            masks=masks, iou_pred=iou_pred, mask_tokens_out=mask_tokens_out,
            object_score_logits=object_score_logits
        )
        return ret

    def _get_stability_scores(self, mask_logits):
        """
        Compute stability scores of the mask logits based on the IoU between upper and
        lower thresholds.
        """
        mask_logits = mask_logits.flatten(-2)
        stability_delta = self.dynamic_multimask_stability_delta
        area_i = torch.sum(mask_logits > stability_delta, dim=-1).float()
        area_u = torch.sum(mask_logits > -stability_delta, dim=-1).float()
        stability_scores = torch.where(area_u > 0, area_i / area_u, 1.0)
        return stability_scores

    def _dynamic_multimask_via_stability(self, all_mask_logits, all_iou_scores):
        """
        When outputting a single mask, if the stability score from the current single-mask
        output (based on output token 0) falls below a threshold, we instead select from
        multi-mask outputs (based on output token 1~3) the mask with the highest predicted
        IoU score. This is intended to ensure a valid mask for both clicking and tracking.
        """
        # The best mask from multimask output tokens (1~3)
        multimask_logits = all_mask_logits[:, 1:, :, :]
        multimask_iou_scores = all_iou_scores[:, 1:]
        best_scores_inds = torch.argmax(multimask_iou_scores, dim=-1)
        batch_inds = torch.arange(
            multimask_iou_scores.size(0), device=all_iou_scores.device
        )
        best_multimask_logits = multimask_logits[batch_inds, best_scores_inds]
        best_multimask_logits = best_multimask_logits.unsqueeze(1)
        best_multimask_iou_scores = multimask_iou_scores[batch_inds, best_scores_inds]
        best_multimask_iou_scores = best_multimask_iou_scores.unsqueeze(1)

        # The mask from singlemask output token 0 and its stability score
        singlemask_logits = all_mask_logits[:, 0:1, :, :]
        singlemask_iou_scores = all_iou_scores[:, 0:1]
        stability_scores = self._get_stability_scores(singlemask_logits)
        is_stable = stability_scores >= self.dynamic_multimask_stability_thresh

        # Dynamically fall back to best multimask output upon low stability scores.
        mask_logits_out = torch.where(
            is_stable[..., None, None].expand_as(singlemask_logits),
            singlemask_logits,
            best_multimask_logits,
        )
        iou_scores_out = torch.where(
            is_stable.expand_as(singlemask_iou_scores),
            singlemask_iou_scores,
            best_multimask_iou_scores,
        )
        return mask_logits_out, iou_scores_out
