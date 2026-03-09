from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, k: int = 3, d: int = 1):
        super().__init__()
        p = d if k == 3 else 0
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=p, dilation=d, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SimpleFPN(nn.Module):
    """轻量 FPN：融合 backbone 多层特征，兼顾上下文和细节。"""

    def __init__(self, in_channels: int, num_levels: int, out_channels: int):
        super().__init__()
        self.laterals = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size=1) for _ in range(num_levels)])
        self.fuses = nn.ModuleList([ConvBNReLU(out_channels, out_channels, 3) for _ in range(num_levels)])

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        laterals = [l(x) for l, x in zip(self.laterals, feats)]
        x = laterals[-1]
        outs = [None] * len(laterals)
        outs[-1] = self.fuses[-1](x)
        for i in range(len(laterals) - 2, -1, -1):
            x = F.interpolate(x, size=laterals[i].shape[-2:], mode="bilinear", align_corners=False) + laterals[i]
            outs[i] = self.fuses[i](x)

        target_size = outs[0].shape[-2:]
        merged = 0.0
        for o in outs:
            merged = merged + F.interpolate(o, size=target_size, mode="bilinear", align_corners=False)
        merged = merged / float(len(outs))
        return merged


class FloodnessHead(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNReLU(in_channels, hidden, 3),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RouterHead(nn.Module):
    """区域路由头：open-like / urban-like / ambiguous。"""

    def __init__(self, in_channels: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNReLU(in_channels, hidden, 3),
            nn.Conv2d(hidden, 3, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OpenExpertHead(nn.Module):
    """开阔区专家：偏大感受野，关注连续洪水斑块。"""

    def __init__(self, in_channels: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNReLU(in_channels, hidden, 3, d=1),
            ConvBNReLU(hidden, hidden, 3, d=2),
            ConvBNReLU(hidden, hidden, 3, d=3),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UrbanExpertHead(nn.Module):
    """城市专家：偏高分辨局部细节，恢复小斑块。"""

    def __init__(self, in_channels: int, high_res_channels: int, hidden: int = 128):
        super().__init__()
        self.high_proj = nn.Conv2d(high_res_channels, hidden, kernel_size=1)
        self.main_proj = nn.Conv2d(in_channels, hidden, kernel_size=1)
        self.refine = nn.Sequential(
            ConvBNReLU(hidden * 2, hidden, 3),
            ConvBNReLU(hidden, hidden, 3),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, high_res_feat: torch.Tensor) -> torch.Tensor:
        h = self.high_proj(high_res_feat)
        h = F.interpolate(h, size=x.shape[-2:], mode="bilinear", align_corners=False)
        m = self.main_proj(x)
        z = torch.cat([m, h], dim=1)
        return self.refine(z)


class StateMemoryAttention(nn.Module):
    """轻量状态记忆模块。

    借鉴 SAM4D 的 memory-conditioned 思想，但改成 SAR 多状态检索：
    - 输入 query 特征
    - 从状态证据图构建 memory banks
    - 按分支偏好（open/urban）做注意力检索
    """

    STATE_ORDER = [
        "intensity_pre",
        "intensity_post",
        "coherence_pre",
        "coherence_co",
        "delta_intensity",
        "delta_coherence",
    ]

    def __init__(self, channels: int):
        super().__init__()
        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.k_proj = nn.Conv2d(1, channels, kernel_size=1)
        self.v_proj = nn.Conv2d(1, channels, kernel_size=1)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)

        self.open_prior = nn.Parameter(torch.tensor([1.2, 1.2, 0.6, 0.6, 1.0, 0.8], dtype=torch.float32), requires_grad=False)
        self.urban_prior = nn.Parameter(torch.tensor([0.6, 0.6, 1.2, 1.2, 0.8, 1.0], dtype=torch.float32), requires_grad=False)

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: B,C,H,W, mask: C(bool)
        idx = torch.where(mask)[0]
        if idx.numel() == 0:
            return x.new_zeros(x.size(0), 1, x.size(2), x.size(3))
        y = x[:, idx, :, :].mean(dim=1, keepdim=True)
        return y

    def build_state_maps(self, imgs: torch.Tensor, feature_type_ids: torch.Tensor, temporal_role_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """根据通道 role 自动构造状态证据图。"""
        # feature_type: 0 coherence, 1 intensity, 2 delta
        # temporal_role: 0 pre, 1 co, 2 post, 3 pre_minus_co, 4 post_minus_pre
        ftype = feature_type_ids[0]
        trole = temporal_role_ids[0]

        maps = {
            "intensity_pre": self._masked_mean(imgs, (ftype == 1) & (trole == 0)),
            "intensity_post": self._masked_mean(imgs, (ftype == 1) & ((trole == 1) | (trole == 2))),
            "coherence_pre": self._masked_mean(imgs, (ftype == 0) & (trole == 0)),
            "coherence_co": self._masked_mean(imgs, (ftype == 0) & (trole == 1)),
            "delta_intensity": self._masked_mean(imgs, (ftype == 2) & (trole == 4)),
            "delta_coherence": self._masked_mean(imgs, (ftype == 2) & (trole == 3)),
        }

        # 若原始输入不含 delta（如 8ch），自动由 pre/post 差异构造。
        if torch.allclose(maps["delta_intensity"], maps["delta_intensity"] * 0):
            maps["delta_intensity"] = maps["intensity_post"] - maps["intensity_pre"]
        if torch.allclose(maps["delta_coherence"], maps["delta_coherence"] * 0):
            maps["delta_coherence"] = maps["coherence_pre"] - maps["coherence_co"]

        return maps

    def _attend(self, query: torch.Tensor, state_maps: Dict[str, torch.Tensor], prior: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(query)
        k_list, v_list = [], []
        for name in self.STATE_ORDER:
            m = state_maps[name]
            if m.shape[-2:] != q.shape[-2:]:
                raise RuntimeError(
                    f"State map '{name}' shape {m.shape[-2:]} mismatches query shape {q.shape[-2:]}"
                )
            k_list.append(self.k_proj(m))
            v_list.append(self.v_proj(m))

        k = torch.stack(k_list, dim=1)  # B,S,C,H,W
        v = torch.stack(v_list, dim=1)

        logits = (q.unsqueeze(1) * k).sum(dim=2) / (q.shape[1] ** 0.5)  # B,S,H,W
        logits = logits * prior.view(1, -1, 1, 1)
        attn = torch.softmax(logits, dim=1)

        fused = (attn.unsqueeze(2) * v).sum(dim=1)
        return self.out_proj(fused)

    def forward(self, query: torch.Tensor, imgs: torch.Tensor, feature_type_ids: torch.Tensor, temporal_role_ids: torch.Tensor):
        state_maps = self.build_state_maps(imgs, feature_type_ids, temporal_role_ids)
        target_hw = query.shape[-2:]
        state_maps = {
            k: (v if v.shape[-2:] == target_hw else F.interpolate(v, size=target_hw, mode="bilinear", align_corners=False))
            for k, v in state_maps.items()
        }
        open_ctx = self._attend(query, state_maps, self.open_prior.to(query.device))
        urban_ctx = self._attend(query, state_maps, self.urban_prior.to(query.device))
        return open_ctx, urban_ctx, state_maps


class AutoPromptRefiner(nn.Module):
    """自动提示细化模块（无人工点击）。"""

    def __init__(self, in_channels: int):
        super().__init__()
        _ = in_channels  # 预留兼容，当前结构按分支固定输入维度。
        # 开阔区分支：偏边界收缩与误检抑制（BG/Open 两类细化）。
        self.open_refine = nn.Sequential(
            ConvBNReLU(6, 64, 3),
            ConvBNReLU(64, 32, 3),
            nn.Conv2d(32, 2, kernel_size=1),
        )
        # 城市分支：偏小斑块恢复与弱响应增强（BG/Urban 两类细化）。
        self.urban_refine = nn.Sequential(
            ConvBNReLU(7, 64, 3),
            ConvBNReLU(64, 32, 3),
            nn.Conv2d(32, 2, kernel_size=1),
        )

    def forward(
        self,
        final_logits: torch.Tensor,
        floodness_prob: torch.Tensor,
        open_prob: torch.Tensor,
        urban_prob: torch.Tensor,
        state_maps: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        probs = torch.softmax(final_logits, dim=1)
        conf = probs.max(dim=1, keepdim=True)[0]
        target_hw = final_logits.shape[-2:]

        state_maps = {
            k: (v if v.shape[-2:] == target_hw else F.interpolate(v, size=target_hw, mode="bilinear", align_corners=False))
            for k, v in state_maps.items()
        }

        pos_open = ((open_prob > 0.6) & (floodness_prob > 0.5)).float()
        pos_urban = ((urban_prob > 0.4) & (floodness_prob > 0.4)).float()
        neg = (floodness_prob < 0.1).float()

        prompts = {
            "pos_open": pos_open,
            "pos_urban": pos_urban,
            "neg": neg,
            "confidence": conf,
        }

        # open 分支输入：bg/open logits + 提示 + ΔI 证据（更关注大斑块边界一致性）。
        open_in = torch.cat(
            [
                final_logits[:, 0:2],
                conf,
                pos_open,
                neg,
                state_maps["delta_intensity"],
            ],
            dim=1,
        )
        open_delta = self.open_refine(open_in)

        # urban 分支输入：bg/urban logits + 提示 + coherence 证据（更关注碎片化小目标）。
        urban_in = torch.cat(
            [
                final_logits[:, [0, 2]],
                conf,
                pos_urban,
                neg,
                state_maps["delta_coherence"],
                state_maps["coherence_co"],
            ],
            dim=1,
        )
        urban_delta = self.urban_refine(urban_in)

        refined = final_logits.clone()
        # BG 同时受 open/urban 分支反证约束，减少洪水误检。
        refined[:, 0:1] = refined[:, 0:1] + 0.25 * (open_delta[:, 0:1] + urban_delta[:, 0:1])
        # Open：鼓励连续区域，负提示区域执行收缩。
        refined[:, 1:2] = refined[:, 1:2] + 0.35 * open_delta[:, 1:2] - 0.15 * neg
        # Urban：在低置信区域结合 coherence 变化做弱响应增强。
        urban_boost = torch.sigmoid(2.5 * state_maps["delta_coherence"]) * (1.0 - conf)
        refined[:, 2:3] = refined[:, 2:3] + 0.35 * urban_delta[:, 1:2] + 0.2 * urban_boost
        return refined, prompts
