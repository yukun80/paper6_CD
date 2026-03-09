from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone_wrapper import PanopticonBackboneWrapper
from models.modules import (
    AutoPromptRefiner,
    FloodnessHead,
    OpenExpertHead,
    RouterHead,
    SimpleFPN,
    StateMemoryAttention,
    UrbanExpertHead,
)


class RegionAwareHierarchicalPanFlood(nn.Module):
    """区域感知、层次化、双专家洪涝分割模型。"""

    def __init__(self, cfg: Dict):
        super().__init__()
        model_cfg = cfg["model"]

        self.enable_router = bool(model_cfg.get("enable_router", True))
        self.enable_memory = bool(model_cfg.get("enable_memory", False))
        self.enable_prompt_refiner = bool(model_cfg.get("enable_prompt_refiner", False))

        self.backbone = PanopticonBackboneWrapper(
            ckpt_path=model_cfg.get("checkpoint_path", ""),
            block_indices=model_cfg.get("block_indices", [3, 5, 7, 11]),
            use_time_embed=model_cfg.get("use_time_embed", True),
            use_metadata_embed=model_cfg.get("use_metadata_embed", True),
            feature_type_num=model_cfg.get("feature_type_num", 3),
            temporal_role_num=model_cfg.get("temporal_role_num", 5),
            polarization_num=model_cfg.get("polarization_num", 2),
            source_role_num=model_cfg.get("source_role_num", 2),
            source_role_scale=model_cfg.get("source_role_scale", 0.1),
        )

        self.fpn_dim = int(model_cfg.get("fpn_dim", 256))
        self.neck = SimpleFPN(in_channels=768, num_levels=len(model_cfg.get("block_indices", [3, 5, 7, 11])), out_channels=self.fpn_dim)

        self.floodness_head = FloodnessHead(self.fpn_dim, hidden=model_cfg.get("head_hidden", 128))
        self.router_head = RouterHead(self.fpn_dim, hidden=model_cfg.get("head_hidden", 128))
        self.open_expert = OpenExpertHead(self.fpn_dim, hidden=model_cfg.get("head_hidden", 128))
        self.urban_expert = UrbanExpertHead(self.fpn_dim, high_res_channels=768, hidden=model_cfg.get("head_hidden", 128))

        self.state_memory = StateMemoryAttention(self.fpn_dim)
        # final(3) + conf(1) + pos_open(1) + pos_urban(1) + neg(1) + evidence(3)
        self.prompt_refiner = AutoPromptRefiner(in_channels=10)

    @staticmethod
    def _compose_final_probs(
        floodness_logit: torch.Tensor,
        open_logit: torch.Tensor,
        urban_logit: torch.Tensor,
        router_logit: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        pf = torch.sigmoid(floodness_logit)
        p_open = torch.sigmoid(open_logit)
        p_urban = torch.sigmoid(urban_logit)

        router_prob = torch.softmax(router_logit, dim=1)
        w_open = router_prob[:, 0:1] + 0.5 * router_prob[:, 2:3]
        w_urban = router_prob[:, 1:2] + 0.5 * router_prob[:, 2:3]

        e_open = w_open * p_open
        e_urban = w_urban * p_urban
        denom = (e_open + e_urban).clamp(min=1e-6)

        p_open_cond = e_open / denom
        p_urban_cond = e_urban / denom

        p_bg = 1.0 - pf
        p_open_final = pf * p_open_cond
        p_urban_final = pf * p_urban_cond

        final_probs = torch.cat([p_bg, p_open_final, p_urban_final], dim=1)
        final_probs = final_probs / final_probs.sum(dim=1, keepdim=True).clamp(min=1e-6)
        final_logits = torch.log(final_probs.clamp(min=1e-6))
        return {
            "final_probs": final_probs,
            "final_logits": final_logits,
            "router_probs": router_prob,
            "open_prob": p_open,
            "urban_prob": p_urban,
            "floodness_prob": pf,
        }

    def forward(self, x_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        bb = self.backbone(x_dict)
        feats = bb["multilevel_feats"]
        fused = self.neck(feats)

        floodness_logit = self.floodness_head(fused)

        if self.enable_router:
            router_logit = self.router_head(fused)
        else:
            router_logit = torch.zeros(
                (fused.size(0), 3, fused.size(2), fused.size(3)), device=fused.device, dtype=fused.dtype
            )
            router_logit[:, 2] = 1.0

        open_feat = fused
        urban_feat = fused
        state_maps = {
            "intensity_pre": fused.new_zeros((fused.size(0), 1, fused.size(2), fused.size(3))),
            "intensity_post": fused.new_zeros((fused.size(0), 1, fused.size(2), fused.size(3))),
            "coherence_pre": fused.new_zeros((fused.size(0), 1, fused.size(2), fused.size(3))),
            "coherence_co": fused.new_zeros((fused.size(0), 1, fused.size(2), fused.size(3))),
            "delta_intensity": fused.new_zeros((fused.size(0), 1, fused.size(2), fused.size(3))),
            "delta_coherence": fused.new_zeros((fused.size(0), 1, fused.size(2), fused.size(3))),
        }

        if self.enable_memory:
            open_ctx, urban_ctx, state_maps = self.state_memory(
                query=fused,
                imgs=x_dict["imgs"],
                feature_type_ids=x_dict["feature_type_ids"],
                temporal_role_ids=x_dict["temporal_role_ids"],
            )
            open_feat = fused + open_ctx
            urban_feat = fused + urban_ctx

        open_logit = self.open_expert(open_feat)
        urban_logit = self.urban_expert(urban_feat, bb["high_res_feat"])

        composed = self._compose_final_probs(floodness_logit, open_logit, urban_logit, router_logit)
        final_logits = composed["final_logits"]
        prompts = None

        if self.enable_prompt_refiner:
            final_logits, prompts = self.prompt_refiner(
                final_logits=final_logits,
                floodness_prob=composed["floodness_prob"],
                open_prob=composed["open_prob"],
                urban_prob=composed["urban_prob"],
                state_maps=state_maps,
            )
            final_probs = torch.softmax(final_logits, dim=1)
        else:
            final_probs = composed["final_probs"]

        out = {
            "floodness_logits": F.interpolate(
                floodness_logit, size=x_dict["imgs"].shape[-2:], mode="bilinear", align_corners=False
            ),
            "router_logits": F.interpolate(router_logit, size=x_dict["imgs"].shape[-2:], mode="bilinear", align_corners=False),
            "open_logits": F.interpolate(open_logit, size=x_dict["imgs"].shape[-2:], mode="bilinear", align_corners=False),
            "urban_logits": F.interpolate(urban_logit, size=x_dict["imgs"].shape[-2:], mode="bilinear", align_corners=False),
            "final_logits": F.interpolate(final_logits, size=x_dict["imgs"].shape[-2:], mode="bilinear", align_corners=False),
            "final_probs": F.interpolate(final_probs, size=x_dict["imgs"].shape[-2:], mode="bilinear", align_corners=False),
            "router_probs": F.interpolate(composed["router_probs"], size=x_dict["imgs"].shape[-2:], mode="bilinear", align_corners=False),
            "state_maps": {k: F.interpolate(v, size=x_dict["imgs"].shape[-2:], mode="bilinear", align_corners=False) for k, v in state_maps.items()},
            "prompts": prompts,
        }
        return out
