"""Hierarchical PanFlood-Adapter 最小可运行示例。"""

import os

# CPU 最小示例关闭 xformers，走普通 attention 路径。
os.environ.setdefault("XFORMERS_DISABLED", "1")

import torch

from urban_floods_hier.dataset import (
    DEFAULT_FEATURE_TYPE_IDS,
    DEFAULT_POLARIZATION_IDS,
    DEFAULT_TEMPORAL_ROLE_IDS,
)
from urban_floods_hier.losses import HierarchicalPanFloodLoss
from urban_floods_hier.model import HierarchicalPanFloodAdapter


def main():
    torch.manual_seed(42)
    device = torch.device("cpu")

    bsz, chn, h, w = 2, 12, 56, 56
    x = torch.randn(bsz, chn, h, w, device=device)

    # 示例 channel_ids/time_ids；实际训练请按配置传入。
    channel_ids = torch.tensor([2101, 2102, 2201, 2202, -2, -1, -2, -1, 2101, 2102, -2, -1], dtype=torch.long)
    time_ids = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1, 3, 3, 4, 4], dtype=torch.long)
    feature_type_ids = torch.tensor(DEFAULT_FEATURE_TYPE_IDS, dtype=torch.long)
    temporal_role_ids = torch.tensor(DEFAULT_TEMPORAL_ROLE_IDS, dtype=torch.long)
    polarization_ids = torch.tensor(DEFAULT_POLARIZATION_IDS, dtype=torch.long)

    x_dict = {
        "imgs": x,
        "chn_ids": channel_ids.unsqueeze(0).repeat(bsz, 1),
        "time_ids": time_ids.unsqueeze(0).repeat(bsz, 1),
        "feature_type_ids": feature_type_ids.unsqueeze(0).repeat(bsz, 1),
        "temporal_role_ids": temporal_role_ids.unsqueeze(0).repeat(bsz, 1),
        "polarization_ids": polarization_ids.unsqueeze(0).repeat(bsz, 1),
    }

    main_label = torch.randint(0, 3, (bsz, h, w), dtype=torch.long, device=device)
    floodness_label = (main_label > 0).long()
    flood_type_label = torch.full_like(main_label, 255)
    flood_type_label[main_label == 1] = 0
    flood_type_label[main_label == 2] = 1

    # 随机加入 ignore 区域，测试 mask 生效。
    ignore_mask = torch.rand(bsz, h, w, device=device) < 0.05
    main_label = main_label.clone()
    main_label[ignore_mask] = 255
    floodness_label = floodness_label.clone()
    floodness_label[ignore_mask] = 255

    valid_mask = torch.rand(bsz, h, w, device=device) > 0.1

    model = HierarchicalPanFloodAdapter(
        ckpt_path=None,
        block_indices=(3, 5, 7, 11),
        fpn_dim=128,
        head_hidden_channels=64,
        use_time_embed=True,
        use_metadata_embed=True,
    ).to(device)

    criterion = HierarchicalPanFloodLoss(
        ignore_index_main=255,
        ignore_index_floodness=255,
        ignore_index_flood_type=255,
        type_loss_kind="bce",
    ).to(device)

    out = model(x_dict)
    loss_dict = criterion(
        floodness_logits=out["floodness_logits"],
        flood_type_logits=out["flood_type_logits"],
        final_logits=out["final_logits"],
        final_probs=out["final_probs"],
        main_label=main_label,
        floodness_label=floodness_label,
        flood_type_label=flood_type_label,
        valid_mask=valid_mask,
    )

    print("floodness_logits:", tuple(out["floodness_logits"].shape))
    print("flood_type_logits:", tuple(out["flood_type_logits"].shape))
    print("final_probs:", tuple(out["final_probs"].shape))

    # 自检 1：三类概率和约等于 1
    prob_sum = out["final_probs"].sum(dim=1)
    max_err = torch.max(torch.abs(prob_sum - 1.0)).item()
    print(f"prob_sum_max_abs_error={max_err:.6e}")

    # 自检 2：flood_type 背景像素不参与损失（统计有效像素数）。
    valid_type = ((flood_type_label != 255) & valid_mask).sum().item()
    print("valid_type_pixels:", valid_type)

    # 自检 3：冻结/解冻接口可用
    model.freeze_backbone()
    n_backbone_trainable_stage_a = sum(p.requires_grad for p in model.backbone.parameters())
    model.unfreeze_backbone_last_n_blocks(5)
    n_backbone_trainable_stage_b = sum(p.requires_grad for p in model.backbone.parameters())
    print("trainable_backbone_params_stageA:", n_backbone_trainable_stage_a)
    print("trainable_backbone_params_stageB:", n_backbone_trainable_stage_b)

    pretty_loss = {k: float(v.item()) for k, v in loss_dict.items() if torch.is_tensor(v) and v.numel() == 1}
    print("loss_dict:", pretty_loss)


if __name__ == "__main__":
    main()
