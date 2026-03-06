from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from dinov2.models import vision_transformer as vits
from urban_floods_hier.heads import BinarySegHead, SharedUPerNeck


def _extract_state_dict(ckpt_obj):
    """兼容多种 checkpoint 格式并提取 backbone 参数。"""
    if isinstance(ckpt_obj, dict):
        if "teacher" in ckpt_obj and isinstance(ckpt_obj["teacher"], dict):
            return ckpt_obj["teacher"]
        if "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
            model_dict = ckpt_obj["model"]
            if any(k.startswith("teacher.backbone.") for k in model_dict):
                out = {}
                prefix = "teacher.backbone."
                for k, v in model_dict.items():
                    if k.startswith(prefix):
                        out[k[len(prefix) :]] = v
                return out
            return model_dict
        return ckpt_obj
    raise ValueError("Unsupported checkpoint format")


def compose_three_class_probs(
    floodness_logits: torch.Tensor,
    flood_type_logits: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """根据层次头输出重组 BG/FO/FU 三类概率。"""
    if floodness_logits.ndim != 4 or flood_type_logits.ndim != 4:
        raise ValueError("Expected 4D logits tensors")
    if floodness_logits.shape != flood_type_logits.shape:
        raise ValueError(f"Logits shape mismatch: {floodness_logits.shape} vs {flood_type_logits.shape}")

    p_f = torch.sigmoid(floodness_logits)
    p_u = torch.sigmoid(flood_type_logits)

    p_bg = 1.0 - p_f
    p_fo = p_f * (1.0 - p_u)
    p_fu = p_f * p_u
    probs = torch.cat([p_bg, p_fo, p_fu], dim=1)
    probs = probs.clamp(min=eps, max=1.0 - eps)

    # 数值归一化，避免累计误差导致概率和偏离 1。
    probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=eps)
    return probs


class HierarchicalPanFloodAdapter(nn.Module):
    """Hierarchical PanFlood-Adapter: Panopticon backbone + shared neck + dual heads。"""

    def __init__(
        self,
        ckpt_path: Optional[str],
        block_indices: Sequence[int] = (3, 5, 7, 11),
        fpn_dim: int = 256,
        ppm_scales: Sequence[int] = (1, 2, 3, 6),
        head_hidden_channels: int = 128,
        use_time_embed: bool = True,
        use_metadata_embed: bool = True,
        feature_type_num: int = 4,
        temporal_role_num: int = 5,
        polarization_num: int = 2,
    ) -> None:
        super().__init__()
        self.block_indices = list(block_indices)

        pe_args = dict(
            attn_dim=2304,
            chnfus_cfg=dict(
                layer_norm=False,
                attn_cfg=dict(num_heads=16),
                use_time_embed=use_time_embed,
                time_embed_num=temporal_role_num,
                use_metadata_embed=use_metadata_embed,
                feature_type_num=feature_type_num,
                temporal_role_num=temporal_role_num,
                polarization_num=polarization_num,
            ),
        )
        self.backbone = vits.vit_base(
            img_size=518,
            patch_size=14,
            init_values=1.0e-5,
            ffn_layer="mlp",
            block_chunks=0,
            qkv_bias=True,
            proj_bias=True,
            ffn_bias=True,
            num_register_tokens=0,
            embed_layer="PanopticonPE",
            pe_args=pe_args,
        )

        self.shared_neck = SharedUPerNeck(
            in_channels=self.backbone.embed_dim,
            num_levels=len(self.block_indices),
            fpn_dim=fpn_dim,
            ppm_scales=ppm_scales,
        )
        self.floodness_head = BinarySegHead(in_channels=fpn_dim, hidden_channels=head_hidden_channels)
        self.flood_type_head = BinarySegHead(in_channels=fpn_dim, hidden_channels=head_hidden_channels)

        self.load_msg = {"missing": [], "unexpected": []}
        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state_dict = _extract_state_dict(ckpt)
            msg = self.backbone.load_state_dict(state_dict, strict=False)
            self.load_msg = {"missing": msg.missing_keys, "unexpected": msg.unexpected_keys}

    def _collect_backbone_features(self, x_dict: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """提取指定 block 的 patch 特征并还原到空间特征图。"""
        imgs = x_dict["imgs"]
        h, w = imgs.shape[-2:]
        if h % self.backbone.patch_size != 0 or w % self.backbone.patch_size != 0:
            raise ValueError(
                f"Input ({h},{w}) must be divisible by patch size {self.backbone.patch_size}. "
                "Use 252x252 or another divisible size."
            )

        h_patch = h // self.backbone.patch_size
        w_patch = w // self.backbone.patch_size

        x = self.backbone.prepare_tokens_with_masks(x_dict)
        need = set(self.block_indices)
        outputs = {}
        for i, blk in enumerate(self.backbone.blocks):
            x = blk(x)
            if i in need:
                outputs[i] = self.backbone.norm(x)

        features: List[torch.Tensor] = []
        for idx in self.block_indices:
            if idx not in outputs:
                raise RuntimeError(f"Requested block {idx} not produced by backbone")
            out = outputs[idx][:, 1 + self.backbone.num_register_tokens :]
            bsz, n_tokens, dim = out.shape
            if n_tokens != h_patch * w_patch:
                raise RuntimeError(f"Unexpected token count: {n_tokens} vs {h_patch * w_patch}")
            feat = out.transpose(1, 2).reshape(bsz, dim, h_patch, w_patch).contiguous()
            features.append(feat)

        return features

    def forward(self, x_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """前向：backbone -> shared neck -> dual heads -> 3 类重组。"""
        feats = self._collect_backbone_features(x_dict)
        shared = self.shared_neck(feats)

        floodness_logits = self.floodness_head(shared)
        flood_type_logits = self.flood_type_head(shared)

        out_size = x_dict["imgs"].shape[-2:]
        floodness_logits = F.interpolate(floodness_logits, size=out_size, mode="bilinear", align_corners=False)
        flood_type_logits = F.interpolate(flood_type_logits, size=out_size, mode="bilinear", align_corners=False)

        final_probs = compose_three_class_probs(floodness_logits, flood_type_logits)
        final_logits = torch.log(final_probs)

        return {
            "floodness_logits": floodness_logits,
            "flood_type_logits": flood_type_logits,
            "final_probs": final_probs,
            "final_logits": final_logits,
            "shared_features": shared,
        }

    def freeze_backbone(self) -> None:
        """冻结 backbone block，但保留 patch_embed 可训练（Stage A）。"""
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.backbone.patch_embed.parameters():
            p.requires_grad = True
        self.backbone.patch_embed.train(True)

    def set_backbone_trainable(self, trainable: bool) -> None:
        """整体控制 backbone 参数是否可训练。"""
        for p in self.backbone.parameters():
            p.requires_grad = trainable
        self.backbone.train(mode=trainable)

    def unfreeze_backbone_last_n_blocks(self, n: int) -> None:
        """解冻最后 n 个 block，并保留 patch_embed 可训练。"""
        self.freeze_backbone()
        n = int(n)
        if n <= 0:
            return

        total_blocks = len(self.backbone.blocks)
        n = min(n, total_blocks)
        start = total_blocks - n
        for idx in range(start, total_blocks):
            for p in self.backbone.blocks[idx].parameters():
                p.requires_grad = True
        for p in self.backbone.norm.parameters():
            p.requires_grad = True

    def named_metadata_parameters(self) -> Iterable:
        """返回元信息 embedding 参数。"""
        for name, p in self.backbone.patch_embed.named_parameters():
            if any(k in name for k in ["feature_type_embed", "temporal_role_embed", "polarization_embed"]):
                yield f"backbone.patch_embed.{name}", p

    def named_input_adapter_parameters(self) -> Iterable:
        """返回 patch_embed 中除元信息 embedding 外的输入适配参数。"""
        for name, p in self.backbone.patch_embed.named_parameters():
            if any(k in name for k in ["feature_type_embed", "temporal_role_embed", "polarization_embed"]):
                continue
            yield f"backbone.patch_embed.{name}", p

    def named_backbone_parameters(self) -> Iterable:
        """返回 backbone 中非 patch_embed 参数。"""
        for name, p in self.backbone.named_parameters():
            if name.startswith("patch_embed."):
                continue
            yield f"backbone.{name}", p

    def named_neck_parameters(self) -> Iterable:
        return self.shared_neck.named_parameters(prefix="shared_neck")

    def named_head_parameters(self) -> Iterable:
        for name, p in self.floodness_head.named_parameters(prefix="floodness_head"):
            yield name, p
        for name, p in self.flood_type_head.named_parameters(prefix="flood_type_head"):
            yield name, p

    def get_trainable_param_groups(
        self,
        lr_backbone: float,
        lr_metadata: float,
        lr_input_adapter: float,
        lr_neck: float,
        lr_heads: float,
        weight_decay: float = 0.01,
    ) -> List[Dict]:
        """按模块返回参数组，便于 Stage A/B 使用不同学习率。"""

        def _collect(params_iter: Iterable) -> List[nn.Parameter]:
            out = []
            for _, p in params_iter:
                if p.requires_grad:
                    out.append(p)
            return out

        metadata_params = _collect(self.named_metadata_parameters())
        input_adapter_params = _collect(self.named_input_adapter_parameters())
        neck_params = [p for p in self.shared_neck.parameters() if p.requires_grad]
        head_params = [p for p in self.floodness_head.parameters() if p.requires_grad] + [
            p for p in self.flood_type_head.parameters() if p.requires_grad
        ]
        backbone_params = _collect(self.named_backbone_parameters())

        param_groups: List[Dict] = []
        if metadata_params:
            param_groups.append(
                {
                    "name": "metadata_embeddings",
                    "params": metadata_params,
                    "lr": lr_metadata,
                    "weight_decay": weight_decay,
                }
            )
        if input_adapter_params:
            param_groups.append(
                {
                    "name": "input_adapter",
                    "params": input_adapter_params,
                    "lr": lr_input_adapter,
                    "weight_decay": weight_decay,
                }
            )
        if neck_params:
            param_groups.append(
                {
                    "name": "shared_neck",
                    "params": neck_params,
                    "lr": lr_neck,
                    "weight_decay": weight_decay,
                }
            )
        if head_params:
            param_groups.append(
                {
                    "name": "hier_heads",
                    "params": head_params,
                    "lr": lr_heads,
                    "weight_decay": weight_decay,
                }
            )
        if backbone_params:
            param_groups.append(
                {
                    "name": "backbone",
                    "params": backbone_params,
                    "lr": lr_backbone,
                    "weight_decay": weight_decay,
                }
            )

        if not param_groups:
            raise RuntimeError("No trainable parameters found. Check freeze/unfreeze setup.")
        return param_groups
