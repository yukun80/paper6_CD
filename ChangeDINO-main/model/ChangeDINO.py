from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .blocks.fpn import FPN, DsBnRelu
from .blocks.cbam import CBAM
from .blocks.adapter import DINOV3Wrapper, DenseAdapterLite
from .blocks.diffatts import TransformerBlock
from .blocks.deform_cross_attn import DeformableCrossAttentionAlign
from .blocks.topo_router import FloodTopoRouter
from .blocks.refiner import HybridRefiner
from .backbone.mobilenetv2 import mobilenet_v2

DEFAULT_BACKBONE_WEIGHT = "pretrained/efficientnet_b0_ra-3dd342df.pth"


class TimmFeatureBackbone(nn.Module):
    """统一 `timm features_only` 骨干的输出接口，保证返回 list[Tensor]。"""

    def __init__(self, model: nn.Module, channels: list[int], reductions: list[int] | None = None):
        super().__init__()
        self.model = model
        self.channels = channels
        self.reductions = reductions or []

    def forward(self, x):
        features = self.model(x)
        if isinstance(features, tuple):
            features = list(features)
        return features


def _load_local_backbone_weights(backbone, weight_path: str, backbone_name: str) -> None:
    """优先加载本地骨干预训练权重，避免训练入口隐式依赖联网下载。"""
    weight_file = Path(weight_path).expanduser()
    if not weight_file.is_file():
        raise FileNotFoundError(f"backbone weight not found: {weight_file}")

    checkpoint = torch.load(weight_file, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict):
        state_dict = (
            checkpoint.get("state_dict")
            or checkpoint.get("model")
            or checkpoint.get("network")
            or checkpoint
        )
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError(f"unsupported backbone checkpoint type: {type(state_dict)}")

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        clean_key = str(key)
        for prefix in ("module.", "model.", "backbone."):
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix) :]
        cleaned_state_dict[clean_key] = value

    missing, unexpected = backbone.load_state_dict(cleaned_state_dict, strict=False)
    print(f"loaded local backbone weights for {backbone_name}: {weight_file}")
    print(
        f"backbone load summary | missing: {len(missing)} | unexpected: {len(unexpected)}"
    )
    if missing:
        print(f"missing keys sample: {missing[:5]}")
    if unexpected:
        print(f"unexpected keys sample: {unexpected[:5]}")


def _build_timm_backbone(model_name: str, backbone_weight: str | None = None) -> TimmFeatureBackbone:
    timm_backbone = timm.create_model(model_name, pretrained=False, features_only=True)
    feature_info = timm_backbone.feature_info
    channels = list(feature_info.channels())
    reductions = list(feature_info.reduction())
    backbone = TimmFeatureBackbone(timm_backbone, channels, reductions)
    if backbone_weight:
        _load_local_backbone_weights(backbone.model, backbone_weight, model_name)
    return backbone


def get_backbone(backbone_name, backbone_weight=DEFAULT_BACKBONE_WEIGHT):
    if backbone_name == "mobilenetv2":
        backbone = mobilenet_v2(pretrained=True, progress=True)
        backbone.channels = [16, 24, 32, 96, 320]
        backbone.reductions = [2, 4, 8, 16, 32]
    elif backbone_name == "efficientnet_b0":
        if not backbone_weight:
            raise ValueError("efficientnet_b0 requires a local PyTorch --backbone_weight")
        backbone = _build_timm_backbone("efficientnet_b0", backbone_weight=backbone_weight)
    else:
        raise NotImplementedError(
            "BACKBONE [%s] is not implemented! Supported backbones: mobilenetv2, efficientnet_b0\n"
            % backbone_name
        )
    return backbone


class PyramidFeatureFusion(nn.Module):
    def __init__(
        self,
        in_dims=[128, 128, 128],
        dense_dim=1024,
        patch_size=16,
        hidden_dim=256,
    ):
        super().__init__()
        self.in_dims = in_dims
        self.dense_dim = dense_dim
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(DsBnRelu(dim + hidden_dim, dim), CBAM(dim, 8))
                for dim in in_dims
            ]
        )

    def forward(self, feas, ds_feas):
        if len(feas) != len(ds_feas) or len(feas) != len(self.blocks):
            raise ValueError(
                f"PyramidFeatureFusion expects matched feature lengths, got "
                f"{len(feas)} CNN / {len(ds_feas)} DINO / {len(self.blocks)} blocks"
            )

        outs = []
        for feat, dino_feat, block in zip(feas, ds_feas, self.blocks):
            outs.append(block(torch.cat([feat, dino_feat], dim=1)))
        return tuple(outs)


class Encoder(nn.Module):
    def __init__(
        self,
        backbone="efficientnet_b0",
        fpn_channels=128,
        deform_groups=4,
        gamma_mode="SE",
        beta_mode="contextgatedconv",
        dino_arch="auto",
        dino_weight="dinov3/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
        backbone_weight=DEFAULT_BACKBONE_WEIGHT,
        device="cuda",
        extract_ids=None,
        **kwargs,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.backbone = get_backbone(backbone, backbone_weight=backbone_weight)
        self.has_native_p1 = len(self.backbone.channels) == 5
        self.fpn = FPN(
            in_channels=self.backbone.channels if self.has_native_p1 else self.backbone.channels[-4:],
            out_channels=fpn_channels,
            deform_groups=deform_groups,
            gamma_mode=gamma_mode,
            beta_mode=beta_mode,
        )
        dense_out_dim = fpn_channels * 2
        self.dino = DINOV3Wrapper(
            dino_arch=dino_arch, weights_path=dino_weight, device=device, extract_ids=extract_ids
        )
        self.dense_adp = DenseAdapterLite(
            in_dim=self.dino.embed_dim, out_dim=dense_out_dim, bottleneck=fpn_channels // 2
        )
        self.pff = PyramidFeatureFusion(
            in_dims=[fpn_channels] * 3,
            dense_dim=self.dino.embed_dim,
            patch_size=self.dino.patch_size,
            hidden_dim=dense_out_dim,
        )
        self.p1_from_p2 = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, 1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        """
        x : [B, 3, H, W]
        returns : (fea_pyramid, dino_raw_levels)
            fea_pyramid      : tuple of 5 级特征，p1/p2 保持 CNN 主导，
                               p3-p5 为 FPN-DINO 融合特征
            dino_raw_levels  : list of 4 个原始 DINOv3 特征，供 Detector 做
                               浅层 gate / 中层 bridge / 深层 bridge 协同
        """
        fea = self.backbone.forward(x)
        fea = self.fpn(fea if self.has_native_p1 else fea[-4:])

        ds_fea_raw = self.dino(x)
        ds_fea = self.dense_adp(ds_fea_raw)

        if len(fea) == 5:
            p1, p2, p3, p4, p5 = fea
        else:
            p1 = self.p1_from_p2(
                F.interpolate(fea[0], scale_factor=2, mode="bilinear", align_corners=False)
            )
            p2, p3, p4, p5 = fea

        p3, p4, p5 = self.pff((p3, p4, p5), ds_fea[1:])

        return (p1, p2, p3, p4, p5), ds_fea_raw


class FuseGated(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(nn.Conv2d(2 * dim, dim, 1, bias=True), nn.Sigmoid())
        self.mix = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
        )

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        support_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x1 = F.interpolate(x1, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        g = self.gate(torch.cat([x1, x2], dim=1))
        if support_map is None:
            support = torch.ones_like(g)
        else:
            support = F.interpolate(
                support_map, size=x2.shape[-2:], mode="bilinear", align_corners=False
            ).clamp(0.0, 1.0)
            if support.shape[1] == 1:
                support = support.expand(-1, g.shape[1], -1, -1)
            elif support.shape[1] != g.shape[1]:
                raise ValueError(
                    f"support_map channels must be 1 or {g.shape[1]}, got {support.shape[1]}"
                )
        fused = x2 + (g * support) * x1
        return self.mix(fused)


class LocalSupportGate(nn.Module):
    """利用浅层稳定变化证据约束粗尺度语义下传。

    当 coarse feature 试图把整块区域推成前景时，只有在 p2 层局部变化
    也提供支持的区域才允许强注入；tiny prior 仅作为小目标旁路增强。
    """

    def __init__(self, dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = max(dim // 2, 32) if hidden_dim is None else hidden_dim
        self.body = nn.Sequential(
            nn.Conv2d(dim + 1, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1, bias=True),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.constant_(self.body[-1].bias, -2.0)

    def forward(
        self,
        support_feat: torch.Tensor,
        target_size: tuple[int, int],
        prior_map: torch.Tensor | None = None,
        prior_boost: bool = False,
    ) -> torch.Tensor:
        support_feat = F.interpolate(
            support_feat, size=target_size, mode="bilinear", align_corners=False
        )
        if prior_map is None:
            prior = support_feat.new_zeros(
                support_feat.shape[0], 1, target_size[0], target_size[1]
            )
        else:
            prior = F.interpolate(
                prior_map, size=target_size, mode="bilinear", align_corners=False
            ).clamp(0.0, 1.0)
        gate = torch.sigmoid(self.body(torch.cat([support_feat, prior], dim=1)))
        if prior_boost:
            gate = torch.maximum(gate, prior)
        return gate


class ContrastAwareDiff(nn.Module):
    """对比度感知差分模块。

    保留稳定版的 abs-diff + 局部对比度残差主线，只在开启 micro gate 时
    为 P2/P3 增加一条更小感受野的对比度分支，并由风险图决定两种残差的占比。
    """

    def __init__(self, dim: int, pool_size: int = 5, tiny_pool_size: int | None = None):
        super().__init__()
        self.local_pool = nn.AvgPool2d(pool_size, stride=1, padding=pool_size // 2)
        self.contrast_branch = self._build_branch(dim)
        self.gate = SpatialChannelGate(dim)

        self.tiny_local_pool = None
        self.tiny_contrast_branch = None
        self.tiny_gate = None
        if tiny_pool_size is not None and tiny_pool_size != pool_size:
            self.tiny_local_pool = nn.AvgPool2d(
                tiny_pool_size, stride=1, padding=tiny_pool_size // 2
            )
            self.tiny_contrast_branch = self._build_branch(dim)
            self.tiny_gate = SpatialChannelGate(dim)
            nn.init.zeros_(self.tiny_contrast_branch[-1].weight)
            nn.init.zeros_(self.tiny_contrast_branch[-1].bias)

    @staticmethod
    def _build_branch(dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
        )

    def _contrast_residual(
        self,
        aligned_pre: torch.Tensor,
        post: torch.Tensor,
        local_pool: nn.Module,
        branch: nn.Module,
        gate: nn.Module,
    ) -> torch.Tensor:
        pre_local = local_pool(aligned_pre)
        post_local = local_pool(post)
        delta_contrast = (post - post_local) - (aligned_pre - pre_local)
        contrast_feat = branch(delta_contrast)
        return gate(contrast_feat) * contrast_feat

    def forward(
        self,
        aligned_pre: torch.Tensor,
        post: torch.Tensor,
        risk_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        abs_diff = torch.abs(aligned_pre - post)
        coarse_residual = self._contrast_residual(
            aligned_pre, post, self.local_pool, self.contrast_branch, self.gate
        )
        if self.tiny_local_pool is None or self.tiny_contrast_branch is None or self.tiny_gate is None:
            return abs_diff + coarse_residual
        if risk_map is None:
            return abs_diff + coarse_residual

        tiny_residual = self._contrast_residual(
            aligned_pre, post, self.tiny_local_pool, self.tiny_contrast_branch, self.tiny_gate
        )
        risk = F.interpolate(
            risk_map, size=abs_diff.shape[-2:], mode="bilinear", align_corners=False
        ).clamp(0.0, 1.0)
        return abs_diff + (1.0 - risk) * coarse_residual + risk * tiny_residual


class SpatialChannelGate(nn.Module):
    """同时保留通道选择与空间显著性，避免 tiny flood 在全局池化中被抹平。"""

    def __init__(self, dim: int):
        super().__init__()
        hidden_dim = max(dim // 4, 8)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden_dim, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, dim, 1),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim, dim, 1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.channel_gate(x) + self.spatial_gate(x)
        return torch.sigmoid(gate)


class DinoTokenBridge(nn.Module):
    """将选定 DINO 层压缩为 token，对 CNN 特征做轻量交叉注意力注入。"""

    def __init__(
        self,
        fpn_dim: int,
        dino_dim: int,
        num_dino_levels: int = 1,
        n_ctx_tokens: int = 64,
        num_heads: int = 4,
    ):
        super().__init__()
        G = int(n_ctx_tokens ** 0.5)
        self.G = G
        self.num_heads = num_heads
        self.head_dim = fpn_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.num_dino_levels = num_dino_levels

        self.dino_proj = nn.Sequential(
            nn.Conv2d(dino_dim * 2 * num_dino_levels, fpn_dim, 1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.SiLU(inplace=True),
        )
        self.q_proj = nn.Conv2d(fpn_dim, fpn_dim, 1, bias=False)
        self.kv_proj = nn.Linear(fpn_dim, fpn_dim * 2, bias=False)
        self.out_proj = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim, 1, bias=False),
            nn.BatchNorm2d(fpn_dim),
        )
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        feat_map: torch.Tensor,
        dino_t1_levels: list[torch.Tensor] | tuple[torch.Tensor, ...],
        dino_t2_levels: list[torch.Tensor] | tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        if len(dino_t1_levels) != self.num_dino_levels or len(dino_t2_levels) != self.num_dino_levels:
            raise ValueError(
                f"DinoTokenBridge expects {self.num_dino_levels} DINO levels, got "
                f"{len(dino_t1_levels)} / {len(dino_t2_levels)}"
            )

        B, C, H, W = feat_map.shape
        dino_ctx = self.dino_proj(torch.cat([*dino_t1_levels, *dino_t2_levels], dim=1))
        tokens = F.adaptive_avg_pool2d(dino_ctx, (self.G, self.G))
        tokens = tokens.flatten(2).permute(0, 2, 1)

        q = self.q_proj(feat_map).flatten(2).permute(0, 2, 1)
        q = q.view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        kv = self.kv_proj(tokens)
        k, v = kv.chunk(2, dim=-1)
        k = k.view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).permute(0, 2, 1, 3).contiguous()
        out = out.view(B, H * W, C).permute(0, 2, 1).view(B, C, H, W)
        out = self.out_proj(out)
        return feat_map + torch.tanh(self.gate) * out


class P1DinoSemanticGate(nn.Module):
    """浅层 DINO 仅提供语义门控，不直接替代 p1 局部细节。"""

    def __init__(self, feat_dim: int, dino_dim: int, num_dino_levels: int = 2):
        super().__init__()
        self.num_dino_levels = num_dino_levels
        self.dino_proj = nn.Sequential(
            nn.Conv2d(dino_dim * 2 * num_dino_levels, feat_dim, 1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.SiLU(inplace=True),
        )
        self.spatial_head = nn.Conv2d(feat_dim, feat_dim, 3, padding=1, bias=True)
        self.channel_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feat_dim, feat_dim, 1, bias=True),
        )
        self.alpha = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.spatial_head.weight)
        nn.init.zeros_(self.spatial_head.bias)
        nn.init.zeros_(self.channel_head[-1].weight)
        nn.init.zeros_(self.channel_head[-1].bias)

    def forward(
        self,
        diff_p1: torch.Tensor,
        dino_t1_levels: list[torch.Tensor] | tuple[torch.Tensor, ...],
        dino_t2_levels: list[torch.Tensor] | tuple[torch.Tensor, ...],
        tiny_prior_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if len(dino_t1_levels) != self.num_dino_levels or len(dino_t2_levels) != self.num_dino_levels:
            raise ValueError(
                f"P1DinoSemanticGate expects {self.num_dino_levels} DINO levels, got "
                f"{len(dino_t1_levels)} / {len(dino_t2_levels)}"
            )

        dino_feat = self.dino_proj(torch.cat([*dino_t1_levels, *dino_t2_levels], dim=1))
        dino_feat = F.interpolate(
            dino_feat, size=diff_p1.shape[-2:], mode="bilinear", align_corners=False
        )
        spatial_gate = torch.tanh(self.spatial_head(dino_feat))
        channel_gate = torch.tanh(self.channel_head(dino_feat))
        gate = spatial_gate * channel_gate
        if tiny_prior_map is not None:
            tiny_prior = F.interpolate(
                tiny_prior_map, size=diff_p1.shape[-2:], mode="bilinear", align_corners=False
            ).clamp(0.0, 1.0)
            gate = gate * (1.0 + tiny_prior)
        return diff_p1 * (1.0 + torch.tanh(self.alpha) * gate)


class DynamicMicroGate(nn.Module):
    """DQ 风格动态微小目标门控。

    结合 p1/p2 的稳定 abs-diff 估计 tiny prior，并为 p2/p3 生成
    风险图，用于调节细粒度对比度增强与 FloodTopoRouter 的修正强度。
    """

    def __init__(self, dim: int, hidden_dim: int = 64):
        super().__init__()
        self.local_pool = nn.AvgPool2d(5, stride=1, padding=2)
        self.tiny_head = nn.Sequential(
            nn.Conv2d(3, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1, bias=True),
        )
        nn.init.zeros_(self.tiny_head[-1].weight)
        nn.init.constant_(self.tiny_head[-1].bias, -1.0)

    def forward(
        self,
        aligned_pre_p1: torch.Tensor,
        post_p1: torch.Tensor,
        abs_diff_p2: torch.Tensor,
        p3_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        abs_diff_p1 = torch.abs(aligned_pre_p1 - post_p1)
        p2_up = F.interpolate(
            abs_diff_p2, size=abs_diff_p1.shape[-2:], mode="bilinear", align_corners=False
        )
        mean_abs_p1 = abs_diff_p1.mean(dim=1, keepdim=True)
        mean_abs_p2_up = p2_up.mean(dim=1, keepdim=True)
        pre_mean = aligned_pre_p1.mean(dim=1, keepdim=True)
        post_mean = post_p1.mean(dim=1, keepdim=True)
        signed_delta_p1 = (post_mean - self.local_pool(post_mean)) - (
            pre_mean - self.local_pool(pre_mean)
        )
        micro_feat = torch.cat(
            [mean_abs_p1, mean_abs_p2_up, signed_delta_p1], dim=1
        )
        tiny_prior_p1 = torch.sigmoid(self.tiny_head(micro_feat))
        risk_p2 = F.interpolate(
            tiny_prior_p1, size=abs_diff_p2.shape[-2:], mode="bilinear", align_corners=False
        )
        risk_p3 = F.interpolate(
            risk_p2, size=p3_size, mode="bilinear", align_corners=False
        )
        return tiny_prior_p1, risk_p2, risk_p3


class NoAlignFeatureAdapter(nn.Module):
    """关闭 soft alignment 时的轻量双时相协同适配器。"""

    def __init__(self, dim: int):
        super().__init__()
        hidden_dim = max(dim // 2, 32)
        self.mix = nn.Sequential(
            nn.Conv2d(dim * 3, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=True),
            nn.Sigmoid(),
        )

        # 以近似 identity 的方式起步，避免关闭 alignment 后训练初期分布突变。
        nn.init.zeros_(self.mix[-1].weight)
        nn.init.zeros_(self.mix[-1].bias)

    def forward(self, pre_feat: torch.Tensor, post_feat: torch.Tensor) -> torch.Tensor:
        if pre_feat.shape != post_feat.shape:
            raise ValueError(
                f"pre/post feature shapes must match, got {pre_feat.shape} vs {post_feat.shape}"
            )
        abs_delta = torch.abs(post_feat - pre_feat)
        residual = self.mix(torch.cat([pre_feat, post_feat, abs_delta], dim=1))
        gate = self.gate(torch.cat([pre_feat, post_feat], dim=1))
        return pre_feat + gate * residual


class Detector(nn.Module):
    def __init__(
        self,
        fpn_channels=128,
        dino_embed_dim=384,
        n_layers=[1, 1, 1, 1],
        disable_soft_alignment=False,
        align_window=5,
        align_points=9,
        align_heads=4,
        align_on_levels=None,
        align_qkv_bias=False,
        align_offset_groups=4,
        contrast_pool_sizes=None,
        p2_window_size=8,
        micro_gate=False,
        dino_collab_mode="multilevel_v2",
        **kwargs,
    ):
        super().__init__()
        if align_on_levels is None:
            align_on_levels = [1, 2, 3]
        if disable_soft_alignment:
            align_on_levels = []
        if contrast_pool_sizes is None:
            contrast_pool_sizes = [5, 5, 5, 5]
        if len(contrast_pool_sizes) != 4:
            raise ValueError(
                f"contrast_pool_sizes expects 4 ints for P2/P3/P4/P5, got {contrast_pool_sizes}"
            )
        invalid_levels = sorted({int(level) for level in align_on_levels if int(level) not in {1, 2, 3}})
        if invalid_levels:
            raise ValueError(f"align_on_levels only supports P1/P2/P3, got {invalid_levels}")
        self.disable_soft_alignment = bool(disable_soft_alignment)
        self.align_on_levels = {int(level) for level in align_on_levels}
        self.use_micro_gate = bool(micro_gate)
        self.p1_pool_size = max(3, contrast_pool_sizes[0])
        self.dino_collab_mode = dino_collab_mode

        self.soft_align_p1 = (
            DeformableCrossAttentionAlign(
                dim=fpn_channels,
                num_heads=max(1, align_heads // 2),
                num_points=max(4, align_points // 2),
                window_size=align_window,
                offset_groups=max(1, align_offset_groups // 2),
                qkv_bias=align_qkv_bias,
            )
            if 1 in self.align_on_levels
            else None
        )
        self.soft_align_p2 = (
            DeformableCrossAttentionAlign(
                dim=fpn_channels,
                num_heads=align_heads,
                num_points=align_points,
                window_size=align_window,
                offset_groups=align_offset_groups,
                qkv_bias=align_qkv_bias,
            )
            if 2 in self.align_on_levels
            else None
        )
        self.soft_align_p3 = (
            DeformableCrossAttentionAlign(
                dim=fpn_channels,
                num_heads=align_heads,
                num_points=align_points,
                window_size=align_window,
                offset_groups=align_offset_groups,
                qkv_bias=align_qkv_bias,
            )
            if 3 in self.align_on_levels
            else None
        )
        use_no_align_adapter = len(self.align_on_levels) == 0
        self.no_align_p1 = NoAlignFeatureAdapter(fpn_channels) if use_no_align_adapter else None
        self.no_align_p2 = NoAlignFeatureAdapter(fpn_channels) if use_no_align_adapter else None
        self.no_align_p3 = NoAlignFeatureAdapter(fpn_channels) if use_no_align_adapter else None
        self.diff_p1 = ContrastAwareDiff(
            fpn_channels,
            pool_size=self.p1_pool_size,
            tiny_pool_size=3 if self.use_micro_gate else None,
        )
        self.diff_p2 = ContrastAwareDiff(
            fpn_channels,
            pool_size=contrast_pool_sizes[0],
            tiny_pool_size=3 if self.use_micro_gate else None,
        )
        self.diff_p3 = ContrastAwareDiff(
            fpn_channels,
            pool_size=contrast_pool_sizes[1],
            tiny_pool_size=3 if self.use_micro_gate else None,
        )
        self.diff_p4 = ContrastAwareDiff(fpn_channels, pool_size=contrast_pool_sizes[2])
        self.diff_p5 = ContrastAwareDiff(fpn_channels, pool_size=contrast_pool_sizes[3])
        self.micro_gate = DynamicMicroGate(fpn_channels) if self.use_micro_gate else None
        self.support_p4 = LocalSupportGate(fpn_channels)
        self.support_p3 = LocalSupportGate(fpn_channels)
        self.support_p2 = LocalSupportGate(fpn_channels)
        self.p5_to_p4 = FuseGated(fpn_channels)
        self.p4_to_p3 = FuseGated(fpn_channels)
        self.p3_to_p2 = FuseGated(fpn_channels)
        self.p2_to_p1 = FuseGated(fpn_channels)

        self.tb5 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=fpn_channels,
                    spatial_attn_type="CDA",
                    num_channel_heads=8,
                    num_spatial_heads=4,
                    depth=3,
                    ffn_expansion_factor=2,
                    bias=False,
                    LayerNorm_type="BiasFree",
                )
                for _ in range(n_layers[0])
            ]
        )
        self.tb4 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=fpn_channels,
                    spatial_attn_type="CDA",
                    num_channel_heads=8,
                    num_spatial_heads=4,
                    depth=3,
                    ffn_expansion_factor=2,
                    bias=False,
                    LayerNorm_type="BiasFree",
                )
                for _ in range(n_layers[1])
            ]
        )
        self.tb3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=fpn_channels,
                    spatial_attn_type="OCDA",
                    window_size=8,
                    overlap_ratio=0.5,
                    num_channel_heads=8,
                    num_spatial_heads=4,
                    depth=2,
                    ffn_expansion_factor=2,
                    bias=False,
                    LayerNorm_type="BiasFree",
                )
                for _ in range(n_layers[2])
            ]
        )
        self.tb2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=fpn_channels,
                    spatial_attn_type="OCDA",
                    window_size=p2_window_size,
                    overlap_ratio=0.5,
                    num_channel_heads=8,
                    num_spatial_heads=4,
                    depth=1,
                    ffn_expansion_factor=2,
                    bias=False,
                    LayerNorm_type="BiasFree",
                )
                for _ in range(n_layers[3])
            ]
        )
        self.tb1 = nn.Sequential(
            DsBnRelu(fpn_channels, fpn_channels),
            CBAM(fpn_channels, 8),
        )
        self.p1_dino_gate = (
            P1DinoSemanticGate(fpn_channels, dino_embed_dim, num_dino_levels=2)
            if self.dino_collab_mode == "multilevel_v2"
            else None
        )
        self.p3_dino_ctx = (
            DinoTokenBridge(fpn_channels, dino_embed_dim, num_dino_levels=2)
            if self.dino_collab_mode == "multilevel_v2"
            else None
        )
        self.dino_ctx = DinoTokenBridge(
            fpn_channels,
            dino_embed_dim,
            num_dino_levels=2 if self.dino_collab_mode == "multilevel_v2" else 1,
        )
        self.p1_head = nn.Conv2d(fpn_channels, 2, 1)
        self.p5_head = nn.Conv2d(fpn_channels, 2, 1)
        self.p4_head = nn.Conv2d(fpn_channels, 2, 1)
        self.p3_head = nn.Conv2d(fpn_channels, 2, 1)
        self.p2_head = nn.Conv2d(fpn_channels, 2, 1)

    @staticmethod
    def _prepare_pre_feat(
        pre_feat: torch.Tensor,
        post_feat: torch.Tensor,
        align_module: nn.Module | None,
        adapter_module: nn.Module | None,
    ) -> torch.Tensor:
        if align_module is not None:
            return align_module(pre_feat, post_feat)
        if adapter_module is not None:
            return adapter_module(pre_feat, post_feat)
        return pre_feat

    def forward(self, x1s, x2s, dino_t1_levels, dino_t2_levels, size=(256, 256)):
        """
        x1s, x2s  : Encoder 返回的 FPN 金字塔特征 (各 5 级)
        dino_t1_levels / dino_t2_levels : Encoder 返回的 4 层 DINOv3 原始特征
        """
        t1_p1, t1_p2, t1_p3, t1_p4, t1_p5 = x1s
        t2_p1, t2_p2, t2_p3, t2_p4, t2_p5 = x2s

        aligned_pre_p1 = self._prepare_pre_feat(
            t1_p1, t2_p1, self.soft_align_p1, self.no_align_p1
        )
        aligned_pre_p2 = self._prepare_pre_feat(
            t1_p2, t2_p2, self.soft_align_p2, self.no_align_p2
        )
        aligned_pre_p3 = self._prepare_pre_feat(
            t1_p3, t2_p3, self.soft_align_p3, self.no_align_p3
        )
        aligned_pre_p4 = t1_p4
        aligned_pre_p5 = t1_p5

        tiny_prior_p1 = None
        risk_p2 = None
        risk_p3 = None
        if self.micro_gate is not None:
            abs_diff_p2 = torch.abs(aligned_pre_p2 - t2_p2)
            tiny_prior_p1, risk_p2, risk_p3 = self.micro_gate(
                aligned_pre_p1, t2_p1, abs_diff_p2, t2_p3.shape[-2:]
            )

        diff_p1 = self.diff_p1(aligned_pre_p1, t2_p1, risk_map=tiny_prior_p1)
        diff_p2 = self.diff_p2(aligned_pre_p2, t2_p2, risk_map=risk_p2)
        diff_p3 = self.diff_p3(aligned_pre_p3, t2_p3, risk_map=risk_p3)
        diff_p4 = self.diff_p4(aligned_pre_p4, t2_p4)
        diff_p5 = self.diff_p5(aligned_pre_p5, t2_p5)
        if self.p1_dino_gate is not None:
            diff_p1 = self.p1_dino_gate(
                diff_p1,
                dino_t1_levels[:2],
                dino_t2_levels[:2],
                tiny_prior_map=tiny_prior_p1,
            )

        # 利用 p2 层较稳定的局部变化证据，抑制粗尺度语义向下游整块扩散。
        support_p4 = self.support_p4(diff_p2, diff_p4.shape[-2:])
        support_p3 = self.support_p3(
            diff_p2, diff_p3.shape[-2:], prior_map=risk_p3, prior_boost=False
        )
        support_p2 = self.support_p2(
            diff_p2, diff_p2.shape[-2:], prior_map=risk_p2, prior_boost=True
        )

        fea_p5 = self.tb5(diff_p5)
        pred_p5 = self.p5_head(fea_p5)
        fea_p4 = self.p5_to_p4(fea_p5, diff_p4, support_map=support_p4)
        fea_p4 = self.tb4(fea_p4)
        pred_p4 = self.p4_head(fea_p4)
        fea_p3 = self.p4_to_p3(fea_p4, diff_p3, support_map=support_p3)
        fea_p3 = self.tb3(fea_p3)
        if self.p3_dino_ctx is not None:
            fea_p3 = self.p3_dino_ctx(fea_p3, dino_t1_levels[1:3], dino_t2_levels[1:3])
        pred_p3 = self.p3_head(fea_p3)
        fea_p2 = self.p3_to_p2(fea_p3, diff_p2, support_map=support_p2)
        fea_p2 = self.tb2(fea_p2)
        if self.dino_collab_mode == "multilevel_v2":
            fea_p2 = self.dino_ctx(fea_p2, dino_t1_levels[2:4], dino_t2_levels[2:4])
        else:
            fea_p2 = self.dino_ctx(fea_p2, [dino_t1_levels[-1]], [dino_t2_levels[-1]])
        pred_p2 = self.p2_head(fea_p2)
        fea_p1 = self.p2_to_p1(fea_p2, diff_p1)
        fea_p1 = self.tb1(fea_p1)
        pred_p1 = self.p1_head(fea_p1)

        pred_p1 = F.interpolate(
            pred_p1, size=size, mode="bilinear", align_corners=False
        )
        pred_p2 = F.interpolate(
            pred_p2, size=size, mode="bilinear", align_corners=False
        )
        pred_p3 = F.interpolate(
            pred_p3, size=size, mode="bilinear", align_corners=False
        )
        pred_p4 = F.interpolate(
            pred_p4, size=size, mode="bilinear", align_corners=False
        )
        pred_p5 = F.interpolate(
            pred_p5, size=size, mode="bilinear", align_corners=False
        )

        return pred_p1, pred_p2, pred_p3, pred_p4, pred_p5, fea_p1, fea_p2, tiny_prior_p1, risk_p2


class ChangeModel(nn.Module):
    def __init__(
        self,
        backbone="efficientnet_b0",
        fpn_channels=128,
        n_layers=[1, 1, 1, 1],
        disable_soft_alignment=False,
        align_window=5,
        align_points=9,
        align_heads=4,
        align_on_levels=None,
        align_qkv_bias=False,
        align_offset_groups=4,
        contrast_pool_sizes=None,
        p2_window_size=8,
        refiner="topo",
        micro_gate=False,
        dino_collab_mode="multilevel_v2",
        branch_consistency_weight=0.05,
        coarse_fp_consistency_weight=0.03,
        consistency_warmup_epochs=15,
        topo_grid_size=16,
        topo_hidden_dim=128,
        topo_neighbor_k=12,
        topo_n_hops=3,
        topo_min_node_occ=0.25,
        topo_neighbor_mode="mixed",
        topo_long_offsets=(2, 4),
        **kwargs,
    ):
        super().__init__()
        self.refiner_mode = refiner
        self.dino_collab_mode = dino_collab_mode
        self.branch_consistency_weight = float(branch_consistency_weight)
        self.coarse_fp_consistency_weight = float(coarse_fp_consistency_weight)
        self.consistency_warmup_epochs = int(consistency_warmup_epochs)
        self.encoder = Encoder(backbone=backbone, fpn_channels=fpn_channels, **kwargs)
        self.detector = Detector(
            fpn_channels=fpn_channels,
            dino_embed_dim=self.encoder.dino.embed_dim,
            n_layers=n_layers,
            disable_soft_alignment=disable_soft_alignment,
            align_window=align_window,
            align_points=align_points,
            align_heads=align_heads,
            align_on_levels=align_on_levels,
            align_qkv_bias=align_qkv_bias,
            align_offset_groups=align_offset_groups,
            contrast_pool_sizes=contrast_pool_sizes,
            p2_window_size=p2_window_size,
            micro_gate=micro_gate,
            dino_collab_mode=dino_collab_mode,
            **kwargs,
        )
        topo_kwargs = {
            "grid_size": topo_grid_size,
            "hidden_dim": topo_hidden_dim,
            "neighbor_k": topo_neighbor_k,
            "n_hops": topo_n_hops,
            "min_node_occ": topo_min_node_occ,
            "neighbor_mode": topo_neighbor_mode,
            "long_offsets": topo_long_offsets,
        }
        if refiner == "topo":
            self.refiner = FloodTopoRouter(feat_dim=fpn_channels, **topo_kwargs)
        elif refiner == "hybrid":
            self.refiner = HybridRefiner(feat_dim=fpn_channels, topo_kwargs=topo_kwargs)
        else:
            raise ValueError(f"Unsupported refiner: {refiner}")

    @staticmethod
    def _branch_consistency_loss(
        pred_p1: torch.Tensor,
        pred_p2: torch.Tensor,
        tiny_prior_map: torch.Tensor | None,
        current_epoch: int | None = None,
        warmup_epochs: int = 0,
    ) -> torch.Tensor:
        if current_epoch is not None and current_epoch <= warmup_epochs:
            return pred_p1.sum() * 0.0
        p1_fg = F.softmax(pred_p1, dim=1)[:, 1:2]
        p2_fg = F.softmax(pred_p2.detach(), dim=1)[:, 1:2]
        if tiny_prior_map is None:
            non_tiny_mask = torch.ones_like(p1_fg)
        else:
            tiny_prior = F.interpolate(
                tiny_prior_map, size=p1_fg.shape[-2:], mode="bilinear", align_corners=False
            ).clamp(0.0, 1.0)
            non_tiny_mask = (tiny_prior < 0.3).float()
        confidence_mask = ((p2_fg - 0.5).abs() > 0.15).float()
        valid_mask = non_tiny_mask * confidence_mask
        if torch.count_nonzero(valid_mask).item() == 0:
            return pred_p1.sum() * 0.0
        diff = F.smooth_l1_loss(p1_fg, p2_fg, reduction="none")
        return (diff * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)

    @staticmethod
    def _coarse_fp_consistency_loss(
        pred_p2: torch.Tensor,
        pred_p4: torch.Tensor,
        pred_p5: torch.Tensor,
        tiny_prior_map: torch.Tensor | None,
        current_epoch: int | None = None,
        warmup_epochs: int = 0,
    ) -> torch.Tensor:
        """抑制 coarse branch 在缺乏局部支持时整块点亮前景。"""
        if current_epoch is not None and current_epoch <= warmup_epochs:
            return pred_p2.sum() * 0.0

        p2_fg = F.softmax(pred_p2.detach(), dim=1)[:, 1:2]
        if tiny_prior_map is None:
            non_tiny_mask = torch.ones_like(p2_fg)
        else:
            tiny_prior = F.interpolate(
                tiny_prior_map, size=p2_fg.shape[-2:], mode="bilinear", align_corners=False
            ).clamp(0.0, 1.0)
            non_tiny_mask = (tiny_prior < 0.3).float()

        weak_support_mask = (p2_fg < 0.35).float()
        valid_mask = non_tiny_mask * weak_support_mask
        if torch.count_nonzero(valid_mask).item() == 0:
            return pred_p2.sum() * 0.0

        loss = pred_p2.sum() * 0.0
        for coarse_pred in (pred_p4, pred_p5):
            coarse_fg = F.softmax(coarse_pred, dim=1)[:, 1:2]
            excess = F.relu(coarse_fg - p2_fg - 0.15)
            loss = loss + (excess * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
        return loss / 2.0

    def _apply_refiner(
        self,
        pred_p1: torch.Tensor,
        pred_p2: torch.Tensor,
        det_p1_feat: torch.Tensor,
        det_p2_feat: torch.Tensor,
        tiny_prior_map: torch.Tensor | None,
        risk_map: torch.Tensor | None,
        gt_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.refiner_mode == "hybrid":
            return self.refiner(
                pred_p1,
                det_p1_feat,
                pred_p2,
                det_p2_feat,
                tiny_prior_map=tiny_prior_map,
                risk_map=risk_map,
                gt_mask=gt_mask,
            )
        return self.refiner(pred_p2, det_p2_feat, risk_map=risk_map, gt_mask=gt_mask)

    @torch.inference_mode()
    def _forward(self, x1, x2):
        fea1, dino_levels1 = self.encoder(x1)
        fea2, dino_levels2 = self.encoder(x2)
        pred_p1, pred_p2, _, _, _, det_p1_feat, det_p2_feat, tiny_prior, risk_map = self.detector(
            fea1, fea2, dino_levels1, dino_levels2, x1.shape[-2:]
        )
        pred, _ = self._apply_refiner(
            pred_p1, pred_p2, det_p1_feat, det_p2_feat, tiny_prior, risk_map
        )
        return pred

    def forward(self, x1, x2, gt_mask=None, current_epoch: int | None = None):
        fea1, dino_levels1 = self.encoder(x1)
        fea2, dino_levels2 = self.encoder(x2)

        pred_p1, pred_p2, pred_p3, pred_p4, pred_p5, det_p1_feat, det_p2_feat, tiny_prior, risk_map = self.detector(
            fea1, fea2, dino_levels1, dino_levels2, x1.shape[-2:]
        )
        final_pred, topo_loss = self._apply_refiner(
            pred_p1, pred_p2, det_p1_feat, det_p2_feat, tiny_prior, risk_map, gt_mask=gt_mask
        )
        consistency_loss = self._branch_consistency_loss(
            pred_p1,
            pred_p2,
            tiny_prior,
            current_epoch=current_epoch,
            warmup_epochs=self.consistency_warmup_epochs,
        )
        coarse_fp_loss = self._coarse_fp_consistency_loss(
            pred_p2,
            pred_p4,
            pred_p5,
            tiny_prior,
            current_epoch=current_epoch,
            warmup_epochs=self.consistency_warmup_epochs,
        )
        return (
            final_pred,
            (pred_p1, pred_p2, pred_p3, pred_p4, pred_p5),
            topo_loss,
            consistency_loss,
            coarse_fp_loss,
        )
