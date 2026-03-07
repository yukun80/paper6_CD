import logging
import json
import math
import os
import warnings
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from pycocotools import mask as coco_mask
from torchsparse.tensor import PointTensor


def _load_sam4d_label(results, filter_cfg=None):
    def decode_mask(encoded_mask):
        """
        Decodes an RLE formatted mask into a binary matrix.

        Args:
            encoded_mask (dict): COCO RLE formatted mask.

        Returns:
            numpy.ndarray: Decoded binary mask matrix.
        """
        if not isinstance(encoded_mask, dict):
            raise ValueError("Encoded mask must be a dictionary.")

        decoded = coco_mask.decode(encoded_mask)
        return decoded

    if filter_cfg is None:
        filter_cfg = {'score': 0, 'volume': 0}

    sam4d_label_path = results['ann_info']['sam4d_label_path']

    assert sam4d_label_path.endswith('.json'), f"{sam4d_label_path} must be a json file"
    with open(sam4d_label_path) as fp:
        sam4d_raw_label = json.load(fp)

    # get obj score statistics
    total_obj_infos = {}
    tgt_sensors = ['LiDAR_TOP'] + results['CAMs']
    for sensor in tgt_sensors:
        if not sensor in sam4d_raw_label['sensors']:
            continue
        for obj_info in sam4d_raw_label['sensors'][sensor]['objects']:
            if obj_info['object_id'] in total_obj_infos:
                continue
            total_obj_infos[obj_info['object_id']] = {
                'volume': obj_info.get('volume', 10),
                'score': obj_info.get('score', 0.5),
                'semantic': obj_info.get('semantic', None)}

    selected_obj_infos = {}
    for obj_id, obj_info in total_obj_infos.items():
        if filter_cfg.get('semantic-white_list', None) is not None and len(filter_cfg['semantic-white_list']) > 0:
            if not np.any([x in obj_info['semantic'] for x in filter_cfg['semantic-white_list']]):
                continue
        if filter_cfg.get('semantic-black_list', None) is not None and len(filter_cfg['semantic-black_list']) > 0:
            if np.any([x in obj_info['semantic'] for x in filter_cfg['semantic-black_list']]):
                continue
        if obj_info['score'] > filter_cfg['score'] and obj_info['volume'] > filter_cfg['volume']:
            selected_obj_infos[obj_id] = obj_info

    if len(selected_obj_infos) == 0:
        warnings.warn(f"sam4d, after filter {filter_cfg}, {sam4d_label_path} does not have any object")
        selected_obj_infos = total_obj_infos

    # get lidar instance mask
    lidar_info = sam4d_raw_label['sensors']['LiDAR_TOP']
    pts_instance_mask, pts_instance_ids = [], []
    pts_ignore_mask = None
    for obj_info in lidar_info['objects']:
        if obj_info['object_id'] == -1:
            pts_ignore_mask = decode_mask(obj_info['lidar_mask']).squeeze(1).astype(np.bool_)
            continue
        if not obj_info['object_id'] in selected_obj_infos:
            continue
        this_mask = decode_mask(obj_info['lidar_mask'])  # (n_pts, 1)
        assert this_mask.shape[0] == results['points'].shape[0]
        if (this_mask == 1).sum() == 0:
            warnings.warn(f"sam4d lidar, obj_id {obj_info['object_id']} in {sam4d_label_path} is empty!!")
            selected_obj_infos.pop(obj_info['object_id'])
            continue
        pts_instance_mask.append(this_mask)
        pts_instance_ids.append(obj_info['object_id'])
    if len(pts_instance_mask) > 0:
        pts_instance_mask = np.concatenate(pts_instance_mask, -1).astype(np.int8)  # (n_pts, n_masks)
        if pts_ignore_mask is None:
            warnings.warn(f"sam4d lidar, {sam4d_label_path} does not have ignore mask (object id -1)")
            pts_ignore_mask = np.zeros(pts_instance_mask.shape[0], dtype=np.bool_)
        pts_instance_mask[pts_ignore_mask] = -1
        results['pts_instance_mask'] = pts_instance_mask
        results['pts_instance_ids'] = np.array(pts_instance_ids, dtype=np.int32)
        assert results['pts_instance_mask'].shape[0] == results['points'].shape[0]

    # get cam instance mask
    imgs_instance_mask, imgs_instance_ids = [], []
    for i, CAM in enumerate(results['CAMs']):
        if not CAM in sam4d_raw_label['sensors']:
            CAM = None
        if CAM is not None:
            cam_info = sam4d_raw_label['sensors'][CAM]
            img_instance_mask, img_instance_ids = [], []
            for obj_info in cam_info['objects']:
                if not obj_info['object_id'] in selected_obj_infos:
                    continue
                this_mask = decode_mask(obj_info['image_mask'])  # (H, W)
                assert (this_mask.shape[0] == results['images'][i].height and this_mask.shape[1] == results['images'][i].width)
                if (this_mask == 1).sum() == 0:
                    warnings.warn(f"sam4d cam, obj_id {obj_info['object_id']} in {sam4d_label_path} is empty!!")
                    selected_obj_infos.pop(obj_info['object_id'])
                    continue
                img_instance_mask.append(this_mask)
                img_instance_ids.append(obj_info['object_id'])
        else:
            img_instance_mask, img_instance_ids = [], []
        if len(img_instance_mask) > 0:
            img_instance_mask = np.stack(img_instance_mask, -1).astype(np.int8)  # (H, W, n_masks)
            img_instance_ids = np.array(img_instance_ids, dtype=np.int32)
        else:
            img_instance_mask = np.zeros((results['images'][i].height, results['images'][i].width, 1), dtype=np.int8)
            img_instance_ids = np.array([0], dtype=np.int32)
        imgs_instance_mask.append(img_instance_mask)
        imgs_instance_ids.append(img_instance_ids)
    results['imgs_instance_mask'] = imgs_instance_mask
    results['imgs_instance_ids'] = imgs_instance_ids

    return results


def get_connected_components(mask):
    """
    Get the connected components (8-connectivity) of binary masks of shape (N, 1, H, W).

    Inputs:
    - mask: A binary mask tensor of shape (N, 1, H, W), where 1 is foreground and 0 is
            background.

    Outputs:
    - labels: A tensor of shape (N, 1, H, W) containing the connected component labels
              for foreground pixels and 0 for background pixels.
    - counts: A tensor of shape (N, 1, H, W) containing the area of the connected
              components for foreground pixels and 0 for background pixels.
    """
    from sam4d.ops.sam2 import cc_ext

    return cc_ext.get_connected_componnets(mask.to(torch.uint8).contiguous())


def fill_holes_in_mask_scores(mask, max_area):
    """
    A post processor to fill small holes in mask scores with area under `max_area`.
    """
    # Holes are those connected components in background with area <= self.max_area
    # (background regions are those with mask scores <= 0)
    assert max_area > 0, "max_area must be positive"

    input_mask = mask
    try:
        labels, areas = get_connected_components(mask <= 0)
        is_hole = (labels > 0) & (areas <= max_area)
        # We fill holes with a small positive mask score (0.1) to change them to foreground.
        mask = torch.where(is_hole, 0.1, mask)
    except Exception as e:
        # Skip the post-processing step on removing small holes if the CUDA kernel fails
        warnings.warn(
            f"{e}\n\nSkipping the post-processing step due to the error above. You can "
            "still use SAM 2 and it's OK to ignore the error above, although some post-processing "
            "functionality may be limited (which doesn't affect the results in most cases; see "
            "https://github.com/facebookresearch/sam2/blob/main/INSTALL.md).",
            category=UserWarning,
            stacklevel=2,
        )
        mask = input_mask

    return mask


def concat_points(old_point_inputs, new_point_inputs):
    """Add new points and labels to previous point inputs (add at the end)."""
    if old_point_inputs is None:
        return new_point_inputs
    else:
        ret = {}
        keys = set(list(old_point_inputs.keys()) + list(new_point_inputs.keys()))
        for key in keys:
            if key not in old_point_inputs:  # 多模态时，如果初次提示为modal A，第二次提示可能为modal B，所以old中可能没有模态B的key
                ret[key] = new_point_inputs[key]
            elif key not in new_point_inputs:  # 同样，new中可能没有模态A的key
                ret[key] = old_point_inputs[key]
            else:
                points = torch.cat([old_point_inputs[key]["point_coords"], new_point_inputs[key]["point_coords"]], dim=1)
                labels = torch.cat([old_point_inputs[key]["point_labels"], new_point_inputs[key]["point_labels"]], dim=1)
                ret[key] = {"point_coords": points, "point_labels": labels}
        return ret


def get_1d_sine_pe(pos_inds, dim, temperature=10000):
    """
    Get 1D sine positional embedding as in the original Transformer paper.
    """
    pe_dim = dim // 2
    dim_t = torch.arange(pe_dim, dtype=torch.float32, device=pos_inds.device)
    dim_t = temperature ** (2 * (dim_t // 2) / pe_dim)

    pos_embed = pos_inds.unsqueeze(-1) / dim_t
    pos_embed = torch.cat([pos_embed.sin(), pos_embed.cos()], dim=-1)
    return pos_embed


def select_closest_cond_frames(frame_idx, cond_frame_outputs, max_cond_frame_num, max_frame_diff):
    """
    Select up to `max_cond_frame_num` conditioning frames from `cond_frame_outputs`
    that are temporally closest to the current frame at `frame_idx`. Here, we take
    - a) the closest conditioning frame before `frame_idx` (if any);
    - b) the closest conditioning frame after `frame_idx` (if any);
    - c) any other temporally closest conditioning frames until reaching a total
         of `max_cond_frame_num` conditioning frames.

    Outputs:
    - selected_outputs: selected items (keys & values) from `cond_frame_outputs`.
    - unselected_outputs: items (keys & values) not selected in `cond_frame_outputs`.
    """
    if max_cond_frame_num == -1 or len(cond_frame_outputs) <= max_cond_frame_num:
        selected_outputs, unselected_outputs = {}, {}
        for f_id, v in cond_frame_outputs.items():
            if abs(frame_idx - f_id) <= max_frame_diff:
                selected_outputs[f_id] = v
            else:
                unselected_outputs[f_id] = v
    else:
        assert max_cond_frame_num >= 2, "we should allow using 2+ conditioning frames"
        selected_outputs = {}

        # the closest conditioning frame before `frame_idx` (if any)
        idx_before = max((t for t in cond_frame_outputs if t < frame_idx), default=None)
        if idx_before is not None and (frame_idx - idx_before) <= max_frame_diff:
            selected_outputs[idx_before] = cond_frame_outputs[idx_before]

        # the closest conditioning frame after `frame_idx` (if any)
        idx_after = min((t for t in cond_frame_outputs if t >= frame_idx), default=None)
        if idx_after is not None and (idx_after - frame_idx) <= max_frame_diff:
            selected_outputs[idx_after] = cond_frame_outputs[idx_after]

        # add other temporally closest conditioning frames until reaching a total
        # of `max_cond_frame_num` conditioning frames.
        num_remain = max_cond_frame_num - len(selected_outputs)
        inds_remain = sorted(
            (t for t in cond_frame_outputs if t not in selected_outputs),
            key=lambda x: abs(x - frame_idx),
        )[:num_remain]
        selected_outputs.update((t, cond_frame_outputs[t]) for t in inds_remain)
        unselected_outputs = {
            t: v for t, v in cond_frame_outputs.items() if t not in selected_outputs
        }

    return selected_outputs, unselected_outputs


def get_next_point(keys, gt_masks, pred_masks, method, pc_points=None):
    if method == "uniform":
        return sample_random_points_from_errors(keys, gt_masks, pred_masks, pc_points=pc_points)
    elif method == "center":
        return sample_one_point_from_error_center(keys, gt_masks, pred_masks, pc_points=pc_points)
    else:
        raise ValueError(f"unknown sampling method {method}")


def sample_random_points_from_errors(keys, gt_masks, pred_masks, num_pt=1, pc_points=None):
    """
    Sample `num_pt` random points (along with their labels) independently of the error regions.

    Inputs:
    - gt_masks: dict 'img' or 'pts', each value is [B, 1, H_im, W_im] masks, dtype=torch.bool
    - pred_masks: dict 'img' or 'pts', each value [B, 1, H_im, W_im] masks, dtype=torch.bool or None
    - num_pt: int, number of points to sample independently for each of the B error maps

    Outputs:
    - points: [B, num_pt, 2], dtype=torch.float, contains (x, y) coordinates of each sampled point
    - labels: [B, num_pt], dtype=torch.int32, where 1 means positive clicks and 0 means
      negative clicks
    """
    assert len(gt_masks) > 0, "gt_masks should not be empty"
    assert isinstance(gt_masks, dict), "gt_masks should be a dict"
    ret = {}
    for key in keys:
        tmp_gt_masks = gt_masks[key]
        tmp_gt_valid = tmp_gt_masks != -1
        tmp_gt_masks = tmp_gt_masks == 1
        if pred_masks is None:  # if pred_masks is not provided, treat it as empty
            tmp_pred_masks = torch.zeros_like(tmp_gt_masks)
        else:
            tmp_pred_masks = pred_masks[key]
        assert tmp_gt_masks.dtype == torch.bool and tmp_gt_masks.size(1) == 1
        assert tmp_pred_masks.dtype == torch.bool and tmp_pred_masks.shape == tmp_gt_masks.shape
        assert num_pt >= 0

        B, _, H_im, W_im = tmp_gt_masks.shape
        device = tmp_gt_masks.device

        # false positive region, a new point sampled in this region should have
        # negative label to correct the FP error
        fp_masks = ~tmp_gt_masks & tmp_pred_masks & tmp_gt_valid
        # false negative region, a new point sampled in this region should have
        # positive label to correct the FN error
        fn_masks = tmp_gt_masks & ~tmp_pred_masks & tmp_gt_valid
        # whether the prediction completely match the ground-truth on each mask
        all_correct = torch.all((tmp_gt_masks == tmp_pred_masks).flatten(2), dim=2)
        all_correct = all_correct[..., None, None]

        # channel 0 is FP map, while channel 1 is FN map
        pts_noise = torch.rand(B, num_pt, H_im, W_im, 2, device=device)
        # sample a negative new click from FP region or a positive new click
        # from FN region, depend on where the maximum falls,
        # and in case the predictions are all correct (no FP or FN), we just
        # sample a negative click from the background region
        pts_noise[..., 0] *= fp_masks | (all_correct & ~tmp_gt_masks)
        pts_noise[..., 1] *= fn_masks
        pts_idx = pts_noise.flatten(2).argmax(dim=2)
        labels = (pts_idx % 2).to(torch.int32)
        pts_idx = pts_idx // 2
        pts_x = pts_idx % W_im
        pts_y = pts_idx // W_im
        if key == 'pts':
            assert pc_points is not None, "pc_points is required when selected_key is pts"
            if isinstance(pc_points, PointTensor):
                pc_points = pc_points.F[:, :3]
            assert pts_x.sum() == 0, "pts_x should be all 0 for pts"
            points = pc_points[pts_y.reshape(-1), :3].view(B, num_pt, 3).to(torch.float)
        else:
            points = torch.stack([pts_x, pts_y], dim=2).to(torch.float)
        ret[key] = {"point_coords": points, "point_labels": labels}
    return ret


def sample_one_point_from_error_center(keys, gt_masks, pred_masks, padding=True, pc_points=None):
    """
    Sample 1 random point (along with its label) from the center of each error region,
    that is, the point with the largest distance to the boundary of each error region.
    This is the RITM sampling method from https://github.com/saic-vul/ritm_interactive_segmentation/blob/master/isegm/inference/clicker.py

    Inputs:
    - gt_masks: [B, 1, H_im, W_im] masks, dtype=torch.bool
    - pred_masks: [B, 1, H_im, W_im] masks, dtype=torch.bool or None
    - padding: if True, pad with boundary of 1 px for distance transform

    Outputs:
    - points: [B, 1, 2], dtype=torch.float, contains (x, y) coordinates of each sampled point
    - labels: [B, 1], dtype=torch.int32, where 1 means positive clicks and 0 means negative clicks
    """
    assert len(gt_masks) > 0, "gt_masks should not be empty"
    assert isinstance(gt_masks, dict), "gt_masks should be a dict"
    ret = {}
    for key in keys:
        tmp_gt_masks = gt_masks[key]
        tmp_gt_valid = tmp_gt_masks != -1
        tmp_gt_masks = tmp_gt_masks == 1
        if pred_masks is None:  # if pred_masks is not provided, treat it as empty
            tmp_pred_masks = torch.zeros_like(tmp_gt_masks)
        else:
            tmp_pred_masks = pred_masks[key]
        assert tmp_gt_masks.dtype == torch.bool and tmp_gt_masks.size(1) == 1  # one mask
        assert tmp_pred_masks.dtype == torch.bool and tmp_pred_masks.shape == tmp_gt_masks.shape

        if key == 'img':
            import cv2
            B, _, _, W_im = tmp_gt_masks.shape
            device = tmp_gt_masks.device

            # false positive region, a new point sampled in this region should have
            # negative label to correct the FP error
            fp_masks = ~tmp_gt_masks & tmp_pred_masks & tmp_gt_valid
            # false negative region, a new point sampled in this region should have
            # positive label to correct the FN error
            fn_masks = tmp_gt_masks & ~tmp_pred_masks & tmp_gt_valid

            fp_masks = fp_masks.cpu().numpy()
            fn_masks = fn_masks.cpu().numpy()
            points = torch.zeros(B, 1, 2, dtype=torch.float)
            labels = torch.ones(B, 1, dtype=torch.int32)
            for b in range(B):
                fn_mask = fn_masks[b, 0]
                fp_mask = fp_masks[b, 0]
                if padding:
                    fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), "constant")
                    fp_mask = np.pad(fp_mask, ((1, 1), (1, 1)), "constant")
                # compute the distance of each point in FN/FP region to its boundary
                fn_mask_dt = cv2.distanceTransform(fn_mask.astype(np.uint8), cv2.DIST_L2, 0)
                fp_mask_dt = cv2.distanceTransform(fp_mask.astype(np.uint8), cv2.DIST_L2, 0)
                if padding:
                    fn_mask_dt = fn_mask_dt[1:-1, 1:-1]
                    fp_mask_dt = fp_mask_dt[1:-1, 1:-1]

                # take the point in FN/FP region with the largest distance to its boundary
                fn_mask_dt_flat = fn_mask_dt.reshape(-1)
                fp_mask_dt_flat = fp_mask_dt.reshape(-1)
                fn_argmax = np.argmax(fn_mask_dt_flat)
                fp_argmax = np.argmax(fp_mask_dt_flat)
                is_positive = fn_mask_dt_flat[fn_argmax] > fp_mask_dt_flat[fp_argmax]
                pt_idx = fn_argmax if is_positive else fp_argmax
                points[b, 0, 0] = pt_idx % W_im  # x
                points[b, 0, 1] = pt_idx // W_im  # y
                labels[b, 0] = int(is_positive)

            points = points.to(device)
            labels = labels.to(device)
            ret[key] = {"point_coords": points, "point_labels": labels}
        else:
            assert key == 'pts'
            B, _, N_pts, _ = tmp_gt_masks.shape
            device = tmp_gt_masks.device

            fp_masks = ~tmp_gt_masks & tmp_pred_masks & tmp_gt_valid
            fn_masks = tmp_gt_masks & ~tmp_pred_masks & tmp_gt_valid

            points = torch.zeros(B, 1, 3, dtype=torch.float)
            labels = torch.ones(B, 1, dtype=torch.int32)

            for b in range(B):
                pts = pc_points.cpu().numpy()  # 确保转换为numpy数组

                # 获取有效区域索引
                fn_idx = np.where(fn_masks[b, 0].cpu().numpy())[0]
                fp_idx = np.where(fp_masks[b, 0].cpu().numpy())[0]

                def find_centroid(points, mask_indices):
                    if len(mask_indices) == 0:
                        return None

                    # 分离掩码内外点云
                    mask_points = points[mask_indices]
                    non_mask_indices = np.setdiff1d(np.arange(len(points)), mask_indices)

                    if len(non_mask_indices) == 0:
                        # 全部点都在掩码内，返回几何中心
                        return np.mean(mask_points, axis=0)

                    # 构建非掩码区域的KD树
                    from scipy.spatial import cKDTree
                    non_mask_tree = cKDTree(points[non_mask_indices])

                    # 计算掩码点到非掩码区域的最小距离
                    distances, _ = non_mask_tree.query(mask_points, k=1)
                    max_idx = np.argmax(distances)

                    return mask_points[max_idx]  # 直接使用掩码点索引

                fn_center = find_centroid(pts, fn_idx)
                fp_center = find_centroid(pts, fp_idx)

                # 决策逻辑优化
                if fn_center is not None and fp_center is not None:
                    fn_size = len(fn_idx)
                    fp_size = len(fp_idx)
                    is_positive = fn_size > fp_size
                elif fn_center is not None:
                    is_positive = True
                elif fp_center is not None:
                    is_positive = False
                else:
                    # 无错误区域时从背景中随机采样一个点
                    bg_mask = ~tmp_gt_masks & tmp_gt_valid
                    if bg_mask.sum() > 0:
                        bg_idx = np.where(bg_mask[b, 0].cpu().numpy())[0]
                        rand_idx = np.random.choice(bg_idx, 1)
                        points[b] = torch.from_numpy(pts[rand_idx]).float()
                        labels[b] = 0
                    else:
                        points[b] = torch.from_numpy(np.zeros((1, 3))).float()
                        labels[b] = 0
                    continue

                target_pt = fn_center if is_positive else fp_center
                points[b] = torch.from_numpy(target_pt).float()
                labels[b] = int(is_positive)

            ret[key] = {
                "point_coords": points.to(device),
                "point_labels": labels.to(device)
            }

    return ret


def sample_box_points(
        keys,
        masks: Dict[str, torch.Tensor],
        noise: float = 0.1,  # SAM default
        noise_bound: Dict[str, int] = {'img': 20, 'pts': 2},  # SAM default
        top_left_label: int = 2,
        bottom_right_label: int = 3,
        pc_points: torch.Tensor = None
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Sample a noised version of the top left and bottom right corners of a given `bbox`

    Inputs:
    - masks: dict, 'pts', 'img', each value is [B, 1, H,W] boxes, dtype=torch.Tensor
    - noise: noise as a fraction of box width and height, dtype=float
    - noise_bound: maximum amount of noise (in pure pixesl), dtype=int

    Returns:
    - box_coords: [B, num_pt, 2], contains (x, y) coordinates of top left and bottom right box corners, dtype=torch.float
    - box_labels: [B, num_pt], label 2 is reserverd for top left and 3 for bottom right corners, dtype=torch.int32
    """
    assert len(masks) > 0, "masks should not be empty"
    assert isinstance(masks, dict), "masks should be a dict"

    ret = {}
    for key in keys:
        tmp_masks = masks[key] == 1

        device = tmp_masks.device
        if key == 'img':
            box_coords = mask_to_box(tmp_masks)
        else:
            assert key == 'pts', f'key should be pts, not {key}'
            assert pc_points is not None
            if isinstance(pc_points, PointTensor):
                pc_points = pc_points.F[:, :3]
            box_coords = pts_mask_to_box(tmp_masks, pc_points)

        B, _, H, W = tmp_masks.shape
        box_labels = torch.tensor([top_left_label, bottom_right_label], dtype=torch.int, device=device).repeat(B)
        if noise > 0.0:
            tmp_noise_bound = noise_bound[key]
            if not isinstance(tmp_noise_bound, torch.Tensor):
                tmp_noise_bound = torch.tensor(tmp_noise_bound, device=device)
            bbox_w = box_coords[..., 2] - box_coords[..., 0]
            bbox_h = box_coords[..., 3] - box_coords[..., 1]
            max_dx = torch.min(bbox_w * noise, tmp_noise_bound)
            max_dy = torch.min(bbox_h * noise, tmp_noise_bound)
            box_noise = 2 * torch.rand(B, 1, 4, device=device) - 1
            box_noise = box_noise * torch.stack((max_dx, max_dy, max_dx, max_dy), dim=-1)

            box_coords = box_coords + box_noise
            if key == 'img':
                img_bounds = torch.tensor([W, H, W, H], device=device) - 1  # uncentered pixel coords
                box_coords.clamp_(torch.zeros_like(img_bounds), img_bounds)  # In place clamping

        box_coords = box_coords.reshape(-1, 2, 2)  # always 2 points
        box_labels = box_labels.reshape(-1, 2)
        if key == 'pts':  # padding z coords
            box_coords = torch.cat([box_coords, torch.zeros(*box_coords.shape[:-1], 1).to(box_coords.device)], dim=-1)
        ret[key] = {"point_coords": box_coords, "point_labels": box_labels}
    return ret


def mask_to_box(masks: torch.Tensor):
    """
    compute bounding box given an input mask

    Inputs:
    - masks: [B, 1, H, W] masks, dtype=torch.Tensor

    Returns:
    - box_coords: [B, 1, 4], contains (x, y) coordinates of top left and bottom right box corners, dtype=torch.Tensor
    """
    B, _, h, w = masks.shape
    device = masks.device
    xs = torch.arange(w, device=device, dtype=torch.int32)
    ys = torch.arange(h, device=device, dtype=torch.int32)
    grid_xs, grid_ys = torch.meshgrid(xs, ys, indexing="xy")
    grid_xs = grid_xs[None, None, ...].expand(B, 1, h, w)
    grid_ys = grid_ys[None, None, ...].expand(B, 1, h, w)
    min_xs, _ = torch.min(torch.where(masks, grid_xs, w).flatten(-2), dim=-1)
    max_xs, _ = torch.max(torch.where(masks, grid_xs, -1).flatten(-2), dim=-1)
    min_ys, _ = torch.min(torch.where(masks, grid_ys, h).flatten(-2), dim=-1)
    max_ys, _ = torch.max(torch.where(masks, grid_ys, -1).flatten(-2), dim=-1)
    bbox_coords = torch.stack((min_xs, min_ys, max_xs, max_ys), dim=-1)

    return bbox_coords


FAKE_PTS_COORD = 100


def pts_mask_to_box(masks: torch.Tensor, pc_points: torch.Tensor):
    """
    compute bounding box given an input mask

    Inputs:
    - masks: [B, 1, n_pts, 1] masks, dtype=torch.Tensor
    - pc_points: [n_pts, 3] original point clouds dtype=torch.Tensor

    Returns:
    - box_coords: [B, 1, 4], contains (x, y) coordinates of top left and bottom right box corners, dtype=torch.Tensor
    """
    assert masks.shape[2] == pc_points.shape[0]
    bbox_coords = []
    for mask in masks:
        obj_points = pc_points[mask.squeeze(0, 2)]
        if obj_points.shape[0] == 0:
            bbox_coords.append(torch.tensor([FAKE_PTS_COORD, FAKE_PTS_COORD, FAKE_PTS_COORD, FAKE_PTS_COORD]))
        else:
            min_xs, min_ys, _ = obj_points.min(0)[0]
            max_xs, max_ys, _ = obj_points.max(0)[0]
            bbox_coords.append(torch.tensor([min_xs, min_ys, max_xs, max_ys]))

    bbox_coords = torch.stack(bbox_coords, dim=0).unsqueeze(1).to(masks.device)

    return bbox_coords


def _load_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
        missing_keys, unexpected_keys = model.load_state_dict(sd)
        if missing_keys:
            logging.error(missing_keys)
            raise RuntimeError()
        if unexpected_keys:
            logging.error(unexpected_keys)
            raise RuntimeError()
        print("Loaded checkpoint sucessfully")


def get_sdpa_settings():
    if torch.cuda.is_available():
        old_gpu = torch.cuda.get_device_properties(0).major < 7
        # only use Flash Attention on Ampere (8.0) or newer GPUs
        use_flash_attn = torch.cuda.get_device_properties(0).major >= 8
        if not use_flash_attn:
            warnings.warn(
                "Flash Attention is disabled as it requires a GPU with Ampere (8.0) CUDA capability.",
                category=UserWarning,
                stacklevel=2,
            )
        # keep math kernel for PyTorch versions before 2.2 (Flash Attention v2 is only
        # available on PyTorch 2.2+, while Flash Attention v1 cannot handle all cases)
        pytorch_version = tuple(int(v) for v in torch.__version__.split(".")[:2])
        if pytorch_version < (2, 2):
            warnings.warn(
                f"You are using PyTorch {torch.__version__} without Flash Attention v2 support. "
                "Consider upgrading to PyTorch 2.2+ for Flash Attention v2 (which could be faster).",
                category=UserWarning,
                stacklevel=2,
            )
        math_kernel_on = pytorch_version < (2, 2) or not use_flash_attn
    else:
        old_gpu = True
        use_flash_attn = False
        math_kernel_on = True

    return old_gpu, use_flash_attn, math_kernel_on
