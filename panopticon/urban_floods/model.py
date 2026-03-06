from typing import Iterable, List, Sequence

import torch
import torch.nn as nn

from urban_floods.heads import UPerLikeHead
from dinov2.models import vision_transformer as vits


def _extract_state_dict(ckpt_obj):
    """兼容两类权重格式并提取 backbone 可加载参数。

    支持:
    1) teacher 扁平参数字典
    2) 含 model.teacher.backbone.* 前缀的训练 checkpoint
    """
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
    if isinstance(ckpt_obj, dict):
        return ckpt_obj
    raise ValueError("Unsupported checkpoint format")


class PanopticonUrbanSeg(nn.Module):
    """Panopticon backbone + UPerLikeHead 的分割模型封装。"""

    def __init__(
        self,
        ckpt_path: str,
        num_classes: int = 3,
        block_indices: Sequence[int] = (3, 5, 7, 11),
        decode_channels: int = 256,
        use_time_embed: bool = True,
    ) -> None:
        super().__init__()
        self.block_indices = list(block_indices)

        pe_args = dict(
            attn_dim=2304,
            chnfus_cfg=dict(
                layer_norm=False,
                attn_cfg=dict(num_heads=16),
                use_time_embed=use_time_embed,
                time_embed_num=2,
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
        # 仅加载 backbone 参数；新增的 time_embed 等参数允许 missing。
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = _extract_state_dict(ckpt)
        msg = self.backbone.load_state_dict(state_dict, strict=False)
        self.load_msg = {"missing": msg.missing_keys, "unexpected": msg.unexpected_keys}

        self.decode_head = UPerLikeHead(
            in_channels=self.backbone.embed_dim,
            num_levels=len(self.block_indices),
            channels=decode_channels,
            num_classes=num_classes,
        )

    def _collect_backbone_features(self, x_dict: dict) -> List[torch.Tensor]:
        """从指定 transformer blocks 提取 patch 特征并还原为空间特征图。"""
        imgs = x_dict["imgs"]
        h, w = imgs.shape[-2:]
        if h % self.backbone.patch_size != 0 or w % self.backbone.patch_size != 0:
            raise ValueError(
                f"Input ({h},{w}) must be divisible by patch size {self.backbone.patch_size}. "
                "Use 252x252 or another divisible size."
            )
        h_patch = h // self.backbone.patch_size
        w_patch = w // self.backbone.patch_size

        # 先做 patch embedding + cls token + position embedding。
        x = self.backbone.prepare_tokens_with_masks(x_dict)
        need = set(self.block_indices)
        outputs = {}
        for i, blk in enumerate(self.backbone.blocks):
            x = blk(x)
            if i in need:
                outputs[i] = self.backbone.norm(x)

        features: List[torch.Tensor] = []
        for idx in self.block_indices:
            # 去掉 cls/register token，仅保留 patch token。
            out = outputs[idx][:, 1 + self.backbone.num_register_tokens :]  # B, N, C
            bsz, n_tokens, dim = out.shape
            if n_tokens != h_patch * w_patch:
                raise RuntimeError(f"Unexpected token count: {n_tokens} vs {h_patch*w_patch}")
            feat = out.transpose(1, 2).reshape(bsz, dim, h_patch, w_patch).contiguous()
            features.append(feat)
        return features

    def forward(self, x_dict: dict) -> torch.Tensor:
        """前向：提取多层特征并解码为分割 logits。"""
        feats = self._collect_backbone_features(x_dict)
        out_size = x_dict["imgs"].shape[-2:]
        logits = self.decode_head(feats, out_size=out_size)
        return logits

    def set_backbone_trainable(self, trainable: bool) -> None:
        """控制 backbone 冻结/解冻，用于分阶段训练。"""
        for p in self.backbone.parameters():
            p.requires_grad = trainable
        self.backbone.train(mode=trainable)

    def named_backbone_parameters(self) -> Iterable:
        return self.backbone.named_parameters()

    def named_head_parameters(self) -> Iterable:
        return self.decode_head.named_parameters()
