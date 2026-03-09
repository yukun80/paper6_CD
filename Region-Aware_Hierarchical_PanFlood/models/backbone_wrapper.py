import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn


def _ensure_panopticon_path() -> None:
    """将 panopticon 目录加入 `sys.path`，以复用其 backbone 实现。"""
    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[2]
    panopticon_root = repo_root / "panopticon"
    if panopticon_root.exists() and str(panopticon_root) not in sys.path:
        sys.path.insert(0, str(panopticon_root))


_ensure_panopticon_path()
from dinov2.models import vision_transformer as vits  # noqa: E402


class PanopticonBackboneWrapper(nn.Module):
    """Panopticon backbone 封装。

    目标：
    1. 保留原 Panopticon 的任意通道编码能力。
    2. 新增 source role embedding，并注入 token 级表示。
    3. 输出多层特征供分层分割头使用。
    """

    def __init__(
        self,
        ckpt_path: str,
        block_indices: Sequence[int] = (3, 5, 7, 11),
        use_time_embed: bool = True,
        use_metadata_embed: bool = True,
        feature_type_num: int = 3,
        temporal_role_num: int = 5,
        polarization_num: int = 2,
        source_role_num: int = 2,
        source_role_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.block_indices = list(block_indices)
        self.source_role_scale = float(source_role_scale)

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

        self.source_role_embed = nn.Embedding(source_role_num, self.backbone.embed_dim)

        self.load_msg = {"missing": [], "unexpected": []}
        if ckpt_path and os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state_dict = self._extract_state_dict(ckpt)
            msg = self.backbone.load_state_dict(state_dict, strict=False)
            self.load_msg = {"missing": msg.missing_keys, "unexpected": msg.unexpected_keys}

    @staticmethod
    def _extract_state_dict(ckpt_obj):
        """兼容 panopticon 权重格式（teacher 或 model.teacher.backbone.*）。"""
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

    def _prepare_tokens_with_roles(self, x_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """重写 token 准备流程，在 patch token 上注入 source role embedding。"""
        x, h, w = self.backbone.patch_embed(x_dict)

        source_role_ids = x_dict.get("source_role_ids", None)
        if source_role_ids is not None:
            src_emb = self.source_role_embed(source_role_ids.long())  # B,C,D
            # source role 是通道语义，融合后以全局上下文形式注入到每个 patch token。
            src_ctx = src_emb.mean(dim=1, keepdim=True)  # B,1,D
            x = x + self.source_role_scale * src_ctx

        x = torch.cat((self.backbone.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.backbone.interpolate_pos_encoding(x, w, h)

        if self.backbone.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.backbone.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )
        return x

    def _collect_multilevel(self, x_dict: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        imgs = x_dict["imgs"]
        h, w = imgs.shape[-2:]
        if h % self.backbone.patch_size != 0 or w % self.backbone.patch_size != 0:
            raise ValueError(
                f"Input ({h},{w}) must be divisible by patch size {self.backbone.patch_size}."
            )

        h_patch = h // self.backbone.patch_size
        w_patch = w // self.backbone.patch_size

        x = self._prepare_tokens_with_roles(x_dict)
        need = set(self.block_indices)
        outputs = {}
        for i, blk in enumerate(self.backbone.blocks):
            x = blk(x)
            if i in need:
                outputs[i] = self.backbone.norm(x)

        feats: List[torch.Tensor] = []
        for idx in self.block_indices:
            out = outputs[idx][:, 1 + self.backbone.num_register_tokens :]
            bsz, n_tokens, dim = out.shape
            if n_tokens != h_patch * w_patch:
                raise RuntimeError(f"Unexpected token count: {n_tokens} vs {h_patch * w_patch}")
            feat = out.transpose(1, 2).reshape(bsz, dim, h_patch, w_patch).contiguous()
            feats.append(feat)
        return feats

    def forward(self, x_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        feats = self._collect_multilevel(x_dict)
        return {
            "multilevel_feats": feats,
            "last_feat": feats[-1],
            "high_res_feat": feats[0],
        }

    def set_backbone_trainable(self, trainable: bool) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = trainable
        self.backbone.train(mode=trainable)

    def freeze_backbone_keep_adapters(self) -> None:
        """冻结主干，仅保留 patch_embed 和 source_role_embed 可训练。"""
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.backbone.patch_embed.parameters():
            p.requires_grad = True
        for p in self.source_role_embed.parameters():
            p.requires_grad = True

    def unfreeze_last_n_blocks(self, n: int) -> None:
        self.freeze_backbone_keep_adapters()
        n = int(n)
        if n <= 0:
            return
        total_blocks = len(self.backbone.blocks)
        start = max(0, total_blocks - n)
        for idx in range(start, total_blocks):
            for p in self.backbone.blocks[idx].parameters():
                p.requires_grad = True
        for p in self.backbone.norm.parameters():
            p.requires_grad = True

    def named_backbone_parameters(self) -> Iterable:
        for name, p in self.backbone.named_parameters(prefix="backbone"):
            # patch_embed 与 source_role 归入 adapter 组，避免优化器参数重复。
            if name.startswith("backbone.patch_embed"):
                continue
            yield name, p

    def named_adapter_parameters(self) -> Iterable:
        for name, p in self.backbone.patch_embed.named_parameters(prefix="backbone.patch_embed"):
            yield name, p
        for name, p in self.source_role_embed.named_parameters(prefix="source_role_embed"):
            yield name, p
