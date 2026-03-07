import copy
import warnings
from collections import OrderedDict

import numpy as np
import torch

from tqdm import tqdm

from sam4d.sam4d_base import SAM4DBase, NO_OBJ_SCORE
from sam4d.misc import concat_points, fill_holes_in_mask_scores
from mmengine.dataset import Compose
from mmengine.registry import MODELS


@MODELS.register_module()
class SAM4DPredictor(SAM4DBase):
    """The predictor class to handle user interactions and manage inference states."""

    def __init__(
            self,
            fill_hole_area=0,
            # whether to apply non-overlapping constraints on the output object masks
            non_overlap_masks=False,
            # whether to clear non-conditioning memory of the surrounding frames (which may contain outdated information) after adding correction clicks;
            # note that this would only apply to *single-object tracking* unless `clear_non_cond_mem_for_multi_obj` is also set to True)
            clear_non_cond_mem_around_input=False,
            # if `add_all_frames_to_correct_as_cond` is True, we also append to the conditioning frame list any frame that receives a later correction click
            # if `add_all_frames_to_correct_as_cond` is False, we conditioning frame list to only use those initial conditioning frames
            add_all_frames_to_correct_as_cond=False,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.fill_hole_area = fill_hole_area
        self.non_overlap_masks = non_overlap_masks
        self.clear_non_cond_mem_around_input = clear_non_cond_mem_around_input
        self.add_all_frames_to_correct_as_cond = add_all_frames_to_correct_as_cond

    def set_data_pipeline(self, pipeline):
        self.pipeline = Compose(pipeline)  # remove bundle and collect

    def get_num_frames(self, inference_state):
        if 'images' in inference_state and not 'points' in inference_state:
            return len(inference_state["images"])
        elif not 'images' in inference_state and 'points' in inference_state:
            return len(inference_state["points"])
        elif 'images' in inference_state and 'points' in inference_state:
            assert len(inference_state["images"]) == len(inference_state["points"])
            return len(inference_state["images"])
        else:
            raise ValueError("Neither 'images' nor 'points' is in the inference state.")

    def get_data(self, img_paths=None, pts_paths=None, metas=None):
        assert img_paths is not None and pts_paths is not None and metas is not None

        examples = []
        for pts_filename, img_filename, ori_meta in tqdm(zip(pts_paths, img_paths, metas), total=len(pts_paths)):
            data_dict = {}
            data_dict['pts_filename'] = pts_filename
            data_dict['img_filename'] = img_filename
            data_dict = self.pipeline(data_dict)
            meta = copy.deepcopy(ori_meta)
            meta.update(dict(img_aug_matrix=data_dict.pop('img_aug_matrix', None),
                             img_norm_cfg=data_dict.pop('img_norm_cfg', None),
                             lidar_aug_matrix=data_dict.pop('lidar_aug_matrix', np.eye(4))))
            data_dict['metas'] = meta
            examples.append(data_dict)

        assert isinstance(examples[0], dict)
        selected_keys = [key for key in examples[0].keys() if key not in ['pts_filename', 'img_filename']]
        ret = {key: [example[key] for example in examples] for key in selected_keys}
        return ret

    @torch.inference_mode()
    def init_state(
            self,
            img_paths=None,
            pts_paths=None,
            metas=None,
            offload_video_to_cpu=False,
            offload_state_to_cpu=False,
            # async_loading_frames=False,
    ):
        data_dict = self.get_data(img_paths, pts_paths, metas)

        compute_device = self.device  # device of the model
        inference_state = {'metas': data_dict['metas']} if 'metas' in data_dict else {}
        if "img" in self.modal:
            images = data_dict["image"]
            inference_state["images"] = images if offload_video_to_cpu else [x.to(compute_device, non_blocking=True) for x in images]
        if "pts" in self.modal:
            points = data_dict["points"]
            inference_state["points"] = points if offload_video_to_cpu else [x.to(compute_device, non_blocking=True) for x in points]
            pt_coords = data_dict["pt_coords"]
            inference_state["pt_coords"] = pt_coords if offload_video_to_cpu else \
                [x.to(compute_device, non_blocking=True) for x in pt_coords]
        inference_state["num_frames"] = self.get_num_frames(inference_state)
        # the original video height and width, used for resizing final output scores
        inference_state["video_height"] = {i: {} for i in range(inference_state["num_frames"])}
        inference_state["video_width"] = {i: {} for i in range(inference_state["num_frames"])}
        for i in range(inference_state["num_frames"]):
            if 'img' in self.modal:
                inference_state["video_height"][i].update({'img': data_dict["img_ori_shape"][i][1]})
                inference_state["video_width"][i].update({'img': data_dict["img_ori_shape"][i][0]})
            if 'pts' in self.modal:
                inference_state["video_height"][i].update({'pts': data_dict["points"][i].shape[0]})
                inference_state["video_width"][i].update({'pts': 1})
        # whether to offload the video frames to CPU memory
        # turning on this option saves the GPU memory with only a very small overhead
        inference_state["offload_video_to_cpu"] = offload_video_to_cpu
        # whether to offload the inference state to CPU memory
        # turning on this option saves the GPU memory at the cost of a lower tracking fps
        # (e.g. in a test case of 768x768 model, fps dropped from 27 to 24 when tracking one object
        # and from 24 to 21 when tracking two objects)
        inference_state["offload_state_to_cpu"] = offload_state_to_cpu
        inference_state["device"] = compute_device
        if offload_state_to_cpu:
            inference_state["storage_device"] = torch.device("cpu")
        else:
            inference_state["storage_device"] = compute_device
        # inputs on each frame
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        # visual features on a small number of recently visited frames for quick interactions
        inference_state["cached_features"] = {}
        # values that don't change across frames (so we only need to hold one copy of them)
        inference_state["constants"] = {}
        # mapping between client-side object id and model-side object index
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
        inference_state["output_dict_per_obj"] = {}
        # A temporary storage to hold new outputs when user interact with a frame
        # to add clicks or mask (it's merged into "output_dict" before propagation starts)
        inference_state["temp_output_dict_per_obj"] = {}
        # Frames that already holds consolidated outputs from click or mask inputs
        # (we directly use their consolidated outputs during tracking)
        # metadata for each tracking frame (e.g. which direction it's tracked)
        inference_state["frames_tracked_per_obj"] = {}
        # record low_res & high_res mask size
        inference_state["low_res_masks_size"] = {}
        # inference_state["high_res_masks_size"] = {}
        inference_state["psam_info"] = {}
        inference_state["sam_info"] = {}
        # Warm up the visual backbone and cache the image feature on frame 0
        self._get_backbone_feature(inference_state, frame_idx=0, batch_size=1)
        return inference_state

    def _obj_id_to_idx(self, inference_state, obj_id):
        """Map client-side object id to model-side object index."""
        obj_idx = inference_state["obj_id_to_idx"].get(obj_id, None)
        if obj_idx is not None:
            return obj_idx

        # We always allow adding new objects (including after tracking starts).
        allow_new_object = True
        if allow_new_object:
            # get the next object slot
            obj_idx = len(inference_state["obj_id_to_idx"])
            inference_state["obj_id_to_idx"][obj_id] = obj_idx
            inference_state["obj_idx_to_id"][obj_idx] = obj_id
            inference_state["obj_ids"] = list(inference_state["obj_id_to_idx"])
            # set up input and output structures for this object
            inference_state["point_inputs_per_obj"][obj_idx] = {}
            inference_state["mask_inputs_per_obj"][obj_idx] = {}
            inference_state["output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
                "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            }
            inference_state["temp_output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
                "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            }
            inference_state["frames_tracked_per_obj"][obj_idx] = {}
            return obj_idx
        else:
            raise RuntimeError(
                f"Cannot add new object id {obj_id} after tracking starts. "
                f"All existing object ids: {inference_state['obj_ids']}. "
                f"Please call 'reset_state' to restart from scratch."
            )

    def _obj_idx_to_id(self, inference_state, obj_idx):
        """Map model-side object index to client-side object id."""
        return inference_state["obj_idx_to_id"][obj_idx]

    def _get_obj_num(self, inference_state):
        """Get the total number of unique object ids received so far in this session."""
        return len(inference_state["obj_idx_to_id"])

    @torch.inference_mode()
    def add_new_points_or_box(
            self,
            inference_state,
            frame_idx,
            obj_id,
            point_prompts=None,
            clear_old_points=True,
            normalize_coords=True,
            box_prompts=None,
    ):
        """Add new points to a frame."""
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        point_inputs_per_frame = inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = inference_state["mask_inputs_per_obj"][obj_idx]

        if point_prompts is None and box_prompts is None:
            raise ValueError("at least one of points or box must be provided as input")

        if point_prompts is None:
            point_prompts = {}
            for key in box_prompts:
                if key == 'img':
                    point_prompts['img'] = {"point_coords": torch.zeros(1, 0, 2, dtype=torch.float32),
                                            "point_labels": torch.zeros(1, 0, dtype=torch.int32)}
                else:
                    assert key == 'pts', f'key must be img or pts, but got {key}'
                    point_prompts['pts'] = {"point_coords": torch.zeros(1, 0, 3, dtype=torch.float32),
                                            "point_labels": torch.zeros(1, 0, dtype=torch.int32)}

        # If `box` is provided, we add it as the first two points with labels 2 and 3
        # along with the user-provided points (consistent with how SAM 2 is trained).
        if box_prompts is not None:
            if not clear_old_points:
                raise ValueError(
                    "cannot add box without clearing old points, since "
                    "box prompt must be provided before any point prompt "
                    "(please use clear_old_points=True instead)"
                )
            for k in box_prompts:
                point_prompts[k]["point_coords"] = torch.cat([box_prompts[k]["point_coords"], point_prompts[k]["point_coords"]], dim=1)
                point_prompts[k]["point_labels"] = torch.cat([box_prompts[k]["point_labels"], point_prompts[k]["point_labels"]], dim=1)

        if 'img' in point_prompts:
            if normalize_coords:
                video_H = inference_state["video_height"][frame_idx]['img']
                video_W = inference_state["video_width"][frame_idx]['img']
                point_prompts['img']["point_coords"] /= torch.tensor([video_W, video_H]).to(point_prompts['img']["point_coords"].device)
            # scale the (normalized) coordinates by the model's internal image size
            point_prompts['img']["point_coords"] *= self.modal['img']['image_size']
            point_prompts['img']["point_coords"] = torch.floor(point_prompts['img']["point_coords"])
        for k in point_prompts:
            point_prompts[k]["point_coords"] = point_prompts[k]["point_coords"].to(inference_state["device"])
            point_prompts[k]["point_labels"] = point_prompts[k]["point_labels"].to(inference_state["device"])

        new_point_inputs = point_prompts

        if not clear_old_points:
            point_inputs = point_inputs_per_frame.get(frame_idx, None)
        else:
            point_inputs = None
        point_inputs = concat_points(point_inputs, new_point_inputs)

        point_inputs_per_frame[frame_idx] = point_inputs
        mask_inputs_per_frame.pop(frame_idx, None)
        # If this frame hasn't been tracked before, we treat it as an initial conditioning
        # frame, meaning that the inputs points are to generate segments on this frame without
        # using any memory from other frames, like in SAM. Otherwise (if it has been tracked),
        # the input points will be used to correct the already tracked masks.
        obj_frames_tracked = inference_state["frames_tracked_per_obj"][obj_idx]
        is_init_cond_frame = frame_idx not in obj_frames_tracked
        # whether to track in reverse time order
        if is_init_cond_frame:
            reverse = False
        else:
            reverse = obj_frames_tracked[frame_idx]["reverse"]
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
        # Add a frame to conditioning output if it's an initial conditioning frame or
        # if the model sees all frames receiving clicks/mask as conditioning frames.
        is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        # Get any previously predicted mask logits on this object and feed it along with
        # the new clicks into the SAM mask decoder.
        prev_sam_mask_logits = None
        # lookup temporary output dict first, which contains the most recent output
        # (if not found, then lookup conditioning and non-conditioning frame output)
        prev_out = obj_temp_output_dict[storage_key].get(frame_idx)
        if prev_out is None:
            prev_out = obj_output_dict["cond_frame_outputs"].get(frame_idx)
            if prev_out is None:
                prev_out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx)

        if prev_out is not None and prev_out["pred_masks"] is not None:
            device = inference_state["device"]
            prev_sam_mask_logits = {k: v.to(device, non_blocking=True) for k, v in prev_out["pred_masks"].items()}
            # Clamp the scale of prev_sam_mask_logits to avoid rare numerical issues.
            prev_sam_mask_logits = {k: torch.clamp(v, -32.0, 32.0) for k, v in prev_sam_mask_logits.items()}
        current_out, _ = self._run_single_frame_inference(
            inference_state=inference_state,
            output_dict=obj_output_dict,  # run on the slice of a single object
            frame_idx=frame_idx,
            batch_size=1,  # run on the slice of a single object
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=point_inputs,
            mask_inputs=None,
            reverse=reverse,
            # Skip the memory encoder when adding clicks or mask. We execute the memory encoder
            # at the beginning of `propagate_in_video` (after user finalize their clicks). This
            # allows us to enforce non-overlapping constraints on all objects before encoding
            # them into memory.
            run_mem_encoder=False,
            prev_sam_mask_logits=prev_sam_mask_logits,
        )
        # Add the output to the output dict (to be used as future memory)
        obj_temp_output_dict[storage_key][frame_idx] = current_out

        # Resize the output mask to the original video resolution
        obj_ids = inference_state["obj_ids"]
        consolidated_out = self._consolidate_temp_output_across_obj(
            inference_state,
            frame_idx,
            is_cond=is_cond,
            consolidate_at_video_res=True,
        )
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated_out["pred_masks_video_res"], frame_idx,
        )
        return frame_idx, obj_ids, video_res_masks

    def add_new_points(self, *args, **kwargs):
        """Deprecated method. Please use `add_new_points_or_box` instead."""
        return self.add_new_points_or_box(*args, **kwargs)

    @torch.inference_mode()
    def add_new_mask(
            self,
            inference_state,
            frame_idx,
            obj_id,
            mask,
    ):
        """Add new mask to a frame."""
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        point_inputs_per_frame = inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = inference_state["mask_inputs_per_obj"][obj_idx]

        assert isinstance(mask, dict), f'mask must be a dict, got {type(mask)}'

        mask_inputs = {}
        for key, this_mask in mask.items():
            assert isinstance(this_mask, torch.Tensor)
            assert this_mask.dim() == 2, f'{key} got mask shape {this_mask.shape}'

            mask_H, mask_W = this_mask.shape
            mask_inputs_orig = this_mask[None, None]  # add batch and channel dimension
            mask_inputs_orig = mask_inputs_orig.float().to(inference_state["device"])

            # resize the mask if it doesn't match the model's image size
            if key == 'img':
                tgt_size = self.modal[key]['image_size']
                if mask_H != tgt_size or mask_W != tgt_size:
                    this_mask_inputs = torch.nn.functional.interpolate(
                        mask_inputs_orig,
                        size=(tgt_size, tgt_size),
                        align_corners=False,
                        mode="bilinear",
                        antialias=True,  # use antialias for downsampling
                    )
                    this_mask_inputs = (this_mask_inputs >= 0.5).float()
            else:
                this_mask_inputs = mask_inputs_orig

            mask_inputs[key] = this_mask_inputs

        mask_inputs_per_frame[frame_idx] = mask_inputs
        point_inputs_per_frame.pop(frame_idx, None)
        # If this frame hasn't been tracked before, we treat it as an initial conditioning
        # frame, meaning that the inputs points are to generate segments on this frame without
        # using any memory from other frames, like in SAM. Otherwise (if it has been tracked),
        # the input points will be used to correct the already tracked masks.
        obj_frames_tracked = inference_state["frames_tracked_per_obj"][obj_idx]
        is_init_cond_frame = frame_idx not in obj_frames_tracked
        # whether to track in reverse time order
        if is_init_cond_frame:
            reverse = False
        else:
            reverse = obj_frames_tracked[frame_idx]["reverse"]
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
        # Add a frame to conditioning output if it's an initial conditioning frame or
        # if the model sees all frames receiving clicks/mask as conditioning frames.
        is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        current_out, _ = self._run_single_frame_inference(
            inference_state=inference_state,
            output_dict=obj_output_dict,  # run on the slice of a single object
            frame_idx=frame_idx,
            batch_size=1,  # run on the slice of a single object
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=None,
            mask_inputs=mask_inputs,
            reverse=reverse,
            # Skip the memory encoder when adding clicks or mask. We execute the memory encoder
            # at the beginning of `propagate_in_video` (after user finalize their clicks). This
            # allows us to enforce non-overlapping constraints on all objects before encoding
            # them into memory.
            run_mem_encoder=False,
        )
        # Add the output to the output dict (to be used as future memory)
        obj_temp_output_dict[storage_key][frame_idx] = current_out

        # Resize the output mask to the original video resolution
        obj_ids = inference_state["obj_ids"]
        consolidated_out = self._consolidate_temp_output_across_obj(
            inference_state,
            frame_idx,
            is_cond=is_cond,
            consolidate_at_video_res=True,
        )
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated_out["pred_masks_video_res"], frame_idx
        )
        return frame_idx, obj_ids, video_res_masks

    def _get_orig_video_res_output(self, inference_state, any_res_masks, frame_idx):
        """
        Resize the object scores to the original video resolution (video_res_masks)
        and apply non-overlapping constraints for final output.
        """
        device = inference_state["device"]
        video_H = inference_state["video_height"][frame_idx]
        video_W = inference_state["video_width"][frame_idx]
        any_res_masks = {k: v.to(device, non_blocking=True) for k, v in any_res_masks.items()}
        video_res_masks = {}
        for k in any_res_masks:
            if any_res_masks[k].shape[-2:] == (video_H[k], video_W[k]):
                video_res_masks[k] = any_res_masks[k]
            else:
                assert k == 'img', f'got {k}'
                video_res_masks[k] = torch.nn.functional.interpolate(
                    any_res_masks[k],
                    size=(video_H[k], video_W[k]),
                    mode="bilinear",
                    align_corners=False,
                )
        if self.non_overlap_masks:
            video_res_masks = {k: self._apply_non_overlapping_constraints(v) for k, v in video_res_masks.items()}
        return any_res_masks, video_res_masks

    def _consolidate_temp_output_across_obj(
            self,
            inference_state,
            frame_idx,
            is_cond,
            consolidate_at_video_res=False,
    ):
        """
        Consolidate the per-object temporary outputs in `temp_output_dict_per_obj` on
        a frame into a single output for all objects, including
        1) fill any missing objects either from `output_dict_per_obj` (if they exist in
           `output_dict_per_obj` for this frame) or leave them as placeholder values
           (if they don't exist in `output_dict_per_obj` for this frame);
        2) if specified, rerun memory encoder after apply non-overlapping constraints
           on the object scores.
        """
        batch_size = self._get_obj_num(inference_state)
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
        # Optionally, we allow consolidating the temporary outputs at the original
        # video resolution (to provide a better editing experience for mask prompts).
        if consolidate_at_video_res:
            consolidated_H = inference_state["video_height"][frame_idx]
            consolidated_W = inference_state["video_width"][frame_idx]
            consolidated_mask_key = "pred_masks_video_res"
        else:
            consolidated_H = {k: v[0] for k, v in inference_state["low_res_masks_size"][frame_idx].items()}
            consolidated_W = {k: v[1] for k, v in inference_state["low_res_masks_size"][frame_idx].items()}
            consolidated_mask_key = "pred_masks"

        # Initialize `consolidated_out`. Its "maskmem_features" and "maskmem_pos_enc"
        # will be added when rerunning the memory encoder after applying non-overlapping
        # constraints to object scores. Its "pred_masks" are prefilled with a large
        # negative value (NO_OBJ_SCORE) to represent missing objects.
        consolidated_out = {
            consolidated_mask_key: {k: torch.full(
                size=(batch_size, 1, consolidated_H[k], consolidated_W[k]),
                fill_value=NO_OBJ_SCORE,
                dtype=torch.float32,
                device=inference_state["storage_device"],
            ) for k in self.modal},
            # "pred_masks_high_res": {k: torch.full(
            #     size=(batch_size, 1, *list(inference_state["high_res_masks_size"][frame_idx][k])),
            #     fill_value=NO_OBJ_SCORE,
            #     dtype=torch.float32,
            #     device=inference_state["storage_device"],
            # ) for k in self.modal},
        }
        for obj_idx in range(batch_size):
            obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            out = obj_temp_output_dict[storage_key].get(frame_idx, None)
            # If the object doesn't appear in "temp_output_dict_per_obj" on this frame,
            # we fall back and look up its previous output in "output_dict_per_obj".
            # We look up both "cond_frame_outputs" and "non_cond_frame_outputs" in
            # "output_dict_per_obj" to find a previous output for this object.
            if out is None:
                out = obj_output_dict["cond_frame_outputs"].get(frame_idx, None)
            if out is None:
                out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx, None)
            # If the object doesn't appear in "output_dict_per_obj" either, we skip it
            # and leave its mask scores to the default scores (i.e. the NO_OBJ_SCORE
            # placeholder above) and set its object pointer to be a dummy pointer.
            if out is None:
                continue
            # Add the temporary object output mask to consolidated output mask
            consolidated_pred_masks = consolidated_out[consolidated_mask_key]
            for k in self.modal:
                # consolidated_out["pred_masks_high_res"][k][obj_idx: obj_idx + 1] = out["pred_masks_high_res"][k]
                if k == 'pts':
                    obj_mask = out["pred_masks_high_res"][k] if consolidate_at_video_res else out["pred_masks"][k]
                    assert obj_mask.shape[-2:] == consolidated_pred_masks[k].shape[-2:]
                    consolidated_pred_masks[k][obj_idx: obj_idx + 1] = obj_mask
                if k == 'img':  # image cannot use pred_masks_high_res because it may be resized original video size
                    obj_mask = out["pred_masks"][k]
                    if obj_mask.shape[-2:] == consolidated_pred_masks[k].shape[-2:]:
                        consolidated_pred_masks[k][obj_idx: obj_idx + 1] = obj_mask
                    else:
                        # Resize first if temporary object mask has a different resolution
                        resized_obj_mask = torch.nn.functional.interpolate(
                            obj_mask,
                            size=consolidated_pred_masks[k].shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                        consolidated_pred_masks[k][obj_idx: obj_idx + 1] = resized_obj_mask

        return consolidated_out

    @torch.inference_mode()
    def propagate_in_video_preflight(self, inference_state):
        """Prepare inference_state and consolidate temporary outputs before tracking."""
        # Check and make sure that every object has received input points or masks.
        batch_size = self._get_obj_num(inference_state)
        if batch_size == 0:
            raise RuntimeError(
                "No input points or masks are provided for any object; please add inputs first."
            )

        # Consolidate per-object temporary outputs in "temp_output_dict_per_obj" and
        # add them into "output_dict".
        for obj_idx in range(batch_size):
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
            for is_cond in [False, True]:
                # Separately consolidate conditioning and non-conditioning temp outputs
                storage_key = (
                    "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
                )
                # Find all the frames that contain temporary outputs for any objects
                # (these should be the frames that have just received clicks for mask inputs
                # via `add_new_points_or_box` or `add_new_mask`)
                for frame_idx, out in obj_temp_output_dict[storage_key].items():
                    # Run memory encoder on the temporary outputs (if the memory feature is missing)
                    if out["maskmem_features"] is None:
                        maskmem_features = self._run_memory_encoder(
                            inference_state=inference_state,
                            frame_idx=frame_idx,
                            batch_size=1,  # run on the slice of a single object
                            high_res_masks=out['pred_masks_high_res'],
                            object_score_logits=out["object_score_logits"],
                            # these frames are what the user interacted with
                            is_mask_from_pts=True,
                            psam_info=inference_state["psam_info"][frame_idx]
                        )
                        out["maskmem_features"] = maskmem_features

                    obj_output_dict[storage_key][frame_idx] = out
                    if self.clear_non_cond_mem_around_input:
                        # clear non-conditioning memory of the surrounding frames
                        self._clear_obj_non_cond_mem_around_input(
                            inference_state, frame_idx, obj_idx
                        )

                # clear temporary outputs in `temp_output_dict_per_obj`
                obj_temp_output_dict[storage_key].clear()

            # check and make sure that every object has received input points or masks
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            if len(obj_output_dict["cond_frame_outputs"]) == 0:
                obj_id = self._obj_idx_to_id(inference_state, obj_idx)
                raise RuntimeError(
                    f"No input points or masks are provided for object id {obj_id}; please add inputs first."
                )
            # edge case: if an output is added to "cond_frame_outputs", we remove any prior
            # output on the same frame in "non_cond_frame_outputs"
            for frame_idx in obj_output_dict["cond_frame_outputs"]:
                obj_output_dict["non_cond_frame_outputs"].pop(frame_idx, None)

    @torch.inference_mode()
    def propagate_in_video(
            self,
            inference_state,
            start_frame_idx=None,
            max_frame_num_to_track=None,
            reverse=False,
    ):
        """Propagate the input points across frames to track in the entire video."""
        self.propagate_in_video_preflight(inference_state)

        obj_ids = inference_state["obj_ids"]
        num_frames = inference_state["num_frames"]
        batch_size = self._get_obj_num(inference_state)

        # set start index, end index, and processing order
        if start_frame_idx is None:
            # default: start from the earliest frame with input points
            start_frame_idx = \
                min(t for obj_output_dict in inference_state["output_dict_per_obj"].values() for t in obj_output_dict["cond_frame_outputs"])
        if max_frame_num_to_track is None:
            # default: track all the frames in the video
            max_frame_num_to_track = num_frames
        if reverse:
            end_frame_idx = max(start_frame_idx - max_frame_num_to_track, 0)
            if start_frame_idx > 0:
                processing_order = range(start_frame_idx, end_frame_idx - 1, -1)
            else:
                processing_order = []  # skip reverse tracking if starting from frame 0
        else:
            end_frame_idx = min(start_frame_idx + max_frame_num_to_track, num_frames - 1)
            processing_order = range(start_frame_idx, end_frame_idx + 1)

        for frame_idx in tqdm(processing_order, desc=f"{len(obj_ids)} objs propagate in video{'(reverse)' if reverse else ''}"):
            pred_masks_per_obj = [None] * batch_size
            for obj_idx in range(batch_size):
                obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
                # We skip those frames already in consolidated outputs (these are frames
                # that received input clicks or mask). Note that we cannot directly run
                # batched forward on them via `_run_single_frame_inference` because the
                # number of clicks on each object might be different.
                if frame_idx in obj_output_dict["cond_frame_outputs"]:
                    storage_key = "cond_frame_outputs"
                    current_out = obj_output_dict[storage_key][frame_idx]
                    device = inference_state["device"]
                    pred_masks = {k: v.to(device, non_blocking=True) for k, v in current_out["pred_masks"].items()}
                    if self.clear_non_cond_mem_around_input:
                        # clear non-conditioning memory of the surrounding frames
                        self._clear_obj_non_cond_mem_around_input(
                            inference_state, frame_idx, obj_idx
                        )
                else:
                    storage_key = "non_cond_frame_outputs"
                    current_out, pred_masks = self._run_single_frame_inference(
                        inference_state=inference_state,
                        output_dict=obj_output_dict,
                        frame_idx=frame_idx,
                        batch_size=1,  # run on the slice of a single object
                        is_init_cond_frame=False,
                        point_inputs=None,
                        mask_inputs=None,
                        reverse=reverse,
                        run_mem_encoder=True,
                    )
                    obj_output_dict[storage_key][frame_idx] = current_out

                inference_state["frames_tracked_per_obj"][obj_idx][frame_idx] = {"reverse": reverse}
                # pred_masks_per_obj[obj_idx] = pred_masks
                pred_masks_per_obj[obj_idx] = current_out["pred_masks_high_res"]

            # Resize the output mask to the original video resolution (we directly use
            # the mask scores on GPU for output to avoid any CPU conversion in between)
            if len(pred_masks_per_obj) > 1:
                all_pred_masks = {k: torch.cat([d[k] for d in pred_masks_per_obj], dim=0) for k in pred_masks_per_obj[0]}
            else:
                all_pred_masks = pred_masks_per_obj[0]
            _, video_res_masks = self._get_orig_video_res_output(
                inference_state, all_pred_masks, frame_idx
            )
            yield frame_idx, obj_ids, video_res_masks

    @torch.inference_mode()
    def clear_all_prompts_in_frame(
            self, inference_state, frame_idx, obj_id, need_output=True
    ):
        """Remove all input points or mask in a specific frame for a given object."""
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)

        # Clear the conditioning information on the given frame
        inference_state["point_inputs_per_obj"][obj_idx].pop(frame_idx, None)
        inference_state["mask_inputs_per_obj"][obj_idx].pop(frame_idx, None)

        temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
        temp_output_dict_per_obj[obj_idx]["cond_frame_outputs"].pop(frame_idx, None)
        temp_output_dict_per_obj[obj_idx]["non_cond_frame_outputs"].pop(frame_idx, None)

        # Remove the frame's conditioning output (possibly downgrading it to non-conditioning)
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        out = obj_output_dict["cond_frame_outputs"].pop(frame_idx, None)
        if out is not None:
            # The frame is not a conditioning frame anymore since it's not receiving inputs,
            # so we "downgrade" its output (if exists) to a non-conditioning frame output.
            obj_output_dict["non_cond_frame_outputs"][frame_idx] = out
            inference_state["frames_tracked_per_obj"][obj_idx].pop(frame_idx, None)

        if not need_output:
            return
        # Finally, output updated masks per object (after removing the inputs above)
        obj_ids = inference_state["obj_ids"]
        is_cond = any(
            frame_idx in obj_temp_output_dict["cond_frame_outputs"]
            for obj_temp_output_dict in temp_output_dict_per_obj.values()
        )
        consolidated_out = self._consolidate_temp_output_across_obj(
            inference_state,
            frame_idx,
            is_cond=is_cond,
            consolidate_at_video_res=True,
        )
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated_out["pred_masks_video_res"], frame_idx
        )
        return frame_idx, obj_ids, video_res_masks

    @torch.inference_mode()
    def reset_state(self, inference_state):
        """Remove all input points or mask in all frames throughout the video."""
        self._reset_tracking_results(inference_state)
        # Remove all object ids
        inference_state["obj_id_to_idx"].clear()
        inference_state["obj_idx_to_id"].clear()
        inference_state["obj_ids"].clear()
        inference_state["point_inputs_per_obj"].clear()
        inference_state["mask_inputs_per_obj"].clear()
        inference_state["output_dict_per_obj"].clear()
        inference_state["temp_output_dict_per_obj"].clear()
        inference_state["frames_tracked_per_obj"].clear()

    def _reset_tracking_results(self, inference_state):
        """Reset all tracking inputs and results across the videos."""
        for v in inference_state["point_inputs_per_obj"].values():
            v.clear()
        for v in inference_state["mask_inputs_per_obj"].values():
            v.clear()
        for v in inference_state["output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        for v in inference_state["temp_output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        for v in inference_state["frames_tracked_per_obj"].values():
            v.clear()

    def _get_backbone_feature(self, inference_state, frame_idx, batch_size):
        """Compute the image features on a given frame."""
        # Look up in the cache first
        input_dict, backbone_out = inference_state["cached_features"].get(frame_idx, ({}, {}))
        if len(backbone_out) == 0:
            # Cache miss -- we will run inference on a single image
            device = inference_state["device"]
            in_dict = {"seq_len": 1}
            if "metas" in inference_state:
                in_dict["metas"] = [[inference_state["metas"][frame_idx]]]
            if "images" in inference_state:
                in_dict["images"] = [inference_state["images"][frame_idx].to(device).float().unsqueeze(0)]
            if "points" in inference_state:
                in_dict["points"] = [[inference_state["points"][frame_idx].to(device).float()]]
                in_dict["pt_coords"] = [[inference_state["pt_coords"][frame_idx].to(device).float()]]
            input_dict = self.prepare_input_dict(**in_dict)
            backbone_out = self.forward_backbone(input_dict)
            # Cache the most recent frame's feature (for repeated interactions with
            # a frame; we can use an LRU cache for more frames in the future).
            inference_state["cached_features"] = {frame_idx: (input_dict, backbone_out)}
            # inference_state["cached_features"][frame_idx] = (input_dict, backbone_out)  # for quick eval!

        psam_info = dict(pts_sp_tensor_info=[x[0] for x in
                                             backbone_out["pts_sp_tensor_info"]] if "pts_sp_tensor_info" in backbone_out else None,
                         pts_org_feats=backbone_out["pts_org_feats"][0] if "pts_org_feats" in backbone_out else None,
                         pts_pos=backbone_out["pts_pos"][-1][0] if "pts_pos" in backbone_out else None)
        sam_info = {}
        inference_state["psam_info"][frame_idx] = psam_info
        inference_state["sam_info"][frame_idx] = sam_info

        backbone_out, vision_feats, feat_sizes = self._prepare_backbone_features(backbone_out)

        img_ids = [0] * batch_size
        current_vision_feats, current_feat_sizes = {}, {}
        # Retrieve image or point features according to img_ids (if they are already computed).
        for k, v in vision_feats.items():
            current_vision_feats[k] = [[x[idx] for idx in img_ids] for x in v]
            current_vision_feats[k] = [torch.stack(x, dim=1) for x in current_vision_feats[k]]  # all objs must in same image!!
            current_feat_sizes[k] = [x[0] for x in feat_sizes[k]]

        return backbone_out, current_vision_feats, current_feat_sizes

    def _run_single_frame_inference(
            self,
            inference_state,
            output_dict,
            frame_idx,
            batch_size,
            is_init_cond_frame,
            point_inputs,
            mask_inputs,
            reverse,
            run_mem_encoder,
            prev_sam_mask_logits=None,
    ):
        """Run tracking on a single frame based on current inputs and previous memory."""
        # Retrieve correct image features
        backbone_out, current_vision_feats, feat_sizes = self._get_backbone_feature(inference_state, frame_idx, batch_size)
        # point and mask should not appear as input simultaneously on the same frame
        assert point_inputs is None or mask_inputs is None
        current_out = self.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=current_vision_feats,
            feat_sizes=feat_sizes,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            output_dict=output_dict,
            num_frames=inference_state["num_frames"],
            track_in_reverse=reverse,
            run_mem_encoder=run_mem_encoder,
            prev_sam_mask_logits=prev_sam_mask_logits,
            psam_infos=inference_state['psam_info'],
            metas=inference_state['metas'] if 'metas' in inference_state else None,
        )

        inference_state["low_res_masks_size"][frame_idx] = {k: v.shape[-2:] for k, v in current_out["pred_masks"].items()}
        # inference_state["high_res_masks_size"][frame_idx] = {k: v.shape[-2:] for k, v in current_out["pred_masks_high_res"].items()}

        # optionally offload the output to CPU memory to save GPU space
        storage_device = inference_state["storage_device"]
        maskmem_features = current_out["maskmem_features"]
        if maskmem_features is not None:
            maskmem_features = {k: v.to(torch.bfloat16) for k, v in maskmem_features.items()}
            maskmem_features = {k: v.to(storage_device, non_blocking=True) for k, v in maskmem_features.items()}
        pred_masks_gpu = current_out["pred_masks"]
        # potentially fill holes in the predicted masks
        if 'img' in pred_masks_gpu and self.fill_hole_area > 0:
            pred_masks_gpu['img'] = fill_holes_in_mask_scores(pred_masks_gpu['img'], self.fill_hole_area)
        pred_masks = {k: v.to(storage_device, non_blocking=True) for k, v in pred_masks_gpu.items()}
        pred_masks_high_res_gpu = current_out["pred_masks_high_res"]
        pred_masks_high_res = {k: v.to(storage_device, non_blocking=True) for k, v in pred_masks_high_res_gpu.items()}
        # "maskmem_pos_enc" is the same across frames, so we only need to store one copy of it
        # maskmem_pos_enc = self._get_maskmem_pos_enc(inference_state, current_out)
        # object pointer is a small tensor, so we always keep it on GPU memory for fast access
        obj_ptr = current_out["obj_ptr"]
        object_score_logits = current_out["object_score_logits"]
        # make a compact version of this frame's output to reduce the state size
        compact_current_out = {
            "maskmem_features": maskmem_features,
            "pred_masks": pred_masks,
            "pred_masks_high_res": pred_masks_high_res,
            "obj_ptr": obj_ptr,
            "object_score_logits": object_score_logits,
        }
        return compact_current_out, pred_masks_gpu

    def _run_memory_encoder(
            self,
            inference_state,
            frame_idx,
            batch_size,
            high_res_masks,
            object_score_logits,
            is_mask_from_pts,
            psam_info=None,
    ):
        """
        Run the memory encoder on `high_res_masks`. This is usually after applying
        non-overlapping constraints to object scores. Since their scores changed, their
        memory also need to be computed again with the memory encoder.
        """
        # Retrieve correct image features
        _, current_vision_feats, feat_sizes = self._get_backbone_feature(inference_state, frame_idx, batch_size)
        maskmem_features = self._encode_new_memory(
            current_vision_feats=current_vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=high_res_masks,
            object_score_logits=object_score_logits,
            is_mask_from_pts=is_mask_from_pts,
            psam_info=psam_info,
        )

        # optionally offload the output to CPU memory to save GPU space
        storage_device = inference_state["storage_device"]
        maskmem_features = {k: v.to(torch.bfloat16).to(storage_device, non_blocking=True) for k, v in maskmem_features.items()}
        # "maskmem_pos_enc" is the same across frames, so we only need to store one copy of it
        # maskmem_pos_enc = self._get_maskmem_pos_enc(inference_state, {"maskmem_pos_enc": maskmem_pos_enc})
        return maskmem_features

    # def _get_maskmem_pos_enc(self, inference_state, current_out):
    #     """
    #     `maskmem_pos_enc` is the same across frames and objects, so we cache it as
    #     a constant in the inference session to reduce session storage size.
    #     """
    #     model_constants = inference_state["constants"]
    #     # "out_maskmem_pos_enc" should be either a list of tensors or None
    #     out_maskmem_pos_enc = current_out["maskmem_pos_enc"]
    #     if out_maskmem_pos_enc is not None:
    #         if "maskmem_pos_enc" not in model_constants:
    #             assert isinstance(out_maskmem_pos_enc, dict)
    #             # only take the slice for one object, since it's same across objects
    #             maskmem_pos_enc = {k: [x[0:1].clone() for x in xx] for k, xx in out_maskmem_pos_enc.items()}
    #             model_constants["maskmem_pos_enc"] = maskmem_pos_enc
    #         else:
    #             maskmem_pos_enc = model_constants["maskmem_pos_enc"]
    #         # expand the cached maskmem_pos_enc to the actual batch size
    #         batch_size = out_maskmem_pos_enc[0].size(0)
    #         expanded_maskmem_pos_enc = [x.expand(batch_size, -1, -1, -1) for x in maskmem_pos_enc]
    #     else:
    #         expanded_maskmem_pos_enc = None
    #     return expanded_maskmem_pos_enc

    @torch.inference_mode()
    def remove_object(self, inference_state, obj_id, strict=False, need_output=True):
        """
        Remove an object id from the tracking state. If strict is True, we check whether
        the object id actually exists and raise an error if it doesn't exist.
        """
        old_obj_idx_to_rm = inference_state["obj_id_to_idx"].get(obj_id, None)
        updated_frames = []
        # Check whether this object_id to remove actually exists and possibly raise an error.
        if old_obj_idx_to_rm is None:
            if not strict:
                return inference_state["obj_ids"], updated_frames
            raise RuntimeError(
                f"Cannot remove object id {obj_id} as it doesn't exist. "
                f"All existing object ids: {inference_state['obj_ids']}."
            )

        # If this is the only remaining object id, we simply reset the state.
        if len(inference_state["obj_id_to_idx"]) == 1:
            self.reset_state(inference_state)
            return inference_state["obj_ids"], updated_frames

        # There are still remaining objects after removing this object id. In this case,
        # we need to delete the object storage from inference state tensors.
        # Step 0: clear the input on those frames where this object id has point or mask input
        # (note that this step is required as it might downgrade conditioning frames to
        # non-conditioning ones)
        obj_input_frames_inds = set()
        obj_input_frames_inds.update(
            inference_state["point_inputs_per_obj"][old_obj_idx_to_rm]
        )
        obj_input_frames_inds.update(
            inference_state["mask_inputs_per_obj"][old_obj_idx_to_rm]
        )
        for frame_idx in obj_input_frames_inds:
            self.clear_all_prompts_in_frame(
                inference_state, frame_idx, obj_id, need_output=False
            )

        # Step 1: Update the object id mapping (note that it must be done after Step 0,
        # since Step 0 still requires the old object id mappings in inference_state)
        old_obj_ids = inference_state["obj_ids"]
        old_obj_inds = list(range(len(old_obj_ids)))
        remain_old_obj_inds = old_obj_inds.copy()
        remain_old_obj_inds.remove(old_obj_idx_to_rm)
        new_obj_ids = [old_obj_ids[old_idx] for old_idx in remain_old_obj_inds]
        new_obj_inds = list(range(len(new_obj_ids)))
        # build new mappings
        old_idx_to_new_idx = dict(zip(remain_old_obj_inds, new_obj_inds))
        inference_state["obj_id_to_idx"] = dict(zip(new_obj_ids, new_obj_inds))
        inference_state["obj_idx_to_id"] = dict(zip(new_obj_inds, new_obj_ids))
        inference_state["obj_ids"] = new_obj_ids

        # Step 2: For per-object tensor storage, we shift their obj_idx in the dict keys.
        def _map_keys(container):
            new_kvs = []
            for k in old_obj_inds:
                v = container.pop(k)
                if k in old_idx_to_new_idx:
                    new_kvs.append((old_idx_to_new_idx[k], v))
            container.update(new_kvs)

        _map_keys(inference_state["point_inputs_per_obj"])
        _map_keys(inference_state["mask_inputs_per_obj"])
        _map_keys(inference_state["output_dict_per_obj"])
        _map_keys(inference_state["temp_output_dict_per_obj"])
        _map_keys(inference_state["frames_tracked_per_obj"])

        # Step 3: Further collect the outputs on those frames in `obj_input_frames_inds`, which
        # could show an updated mask for objects previously occluded by the object being removed
        if need_output:
            temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
            for frame_idx in obj_input_frames_inds:
                is_cond = any(
                    frame_idx in obj_temp_output_dict["cond_frame_outputs"]
                    for obj_temp_output_dict in temp_output_dict_per_obj.values()
                )
                consolidated_out = self._consolidate_temp_output_across_obj(
                    inference_state,
                    frame_idx,
                    is_cond=is_cond,
                    consolidate_at_video_res=True,
                )
                _, video_res_masks = self._get_orig_video_res_output(
                    inference_state, consolidated_out["pred_masks_video_res"], frame_idx
                )
                updated_frames.append((frame_idx, video_res_masks))

        return inference_state["obj_ids"], updated_frames

    def _clear_non_cond_mem_around_input(self, inference_state, frame_idx):
        """
        Remove the non-conditioning memory around the input frame. When users provide
        correction clicks, the surrounding frames' non-conditioning memories can still
        contain outdated object appearance information and could confuse the model.

        This method clears those non-conditioning memories surrounding the interacted
        frame to avoid giving the model both old and new information about the object.
        """
        r = self.memory_temporal_stride_for_eval
        frame_idx_begin = frame_idx - r * self.num_maskmem
        frame_idx_end = frame_idx + r * self.num_maskmem
        batch_size = self._get_obj_num(inference_state)
        for obj_idx in range(batch_size):
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            non_cond_frame_outputs = obj_output_dict["non_cond_frame_outputs"]
            for t in range(frame_idx_begin, frame_idx_end + 1):
                non_cond_frame_outputs.pop(t, None)
