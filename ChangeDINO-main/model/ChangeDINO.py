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
from .backbone.mobilenetv2 import mobilenet_v2

DEFAULT_CONVNEXTV2_NANO_WEIGHT = "pretrained/convnextv2_nano_22k_224_ema.pt"


class TimmFeatureBackbone(nn.Module):
    """统一 `timm features_only` 骨干的输出接口，保证返回 list[Tensor]。"""

    def __init__(self, model: nn.Module, channels: list[int]):
        super().__init__()
        self.model = model
        self.channels = channels

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


def get_backbone(backbone_name, backbone_weight=DEFAULT_CONVNEXTV2_NANO_WEIGHT):
    if backbone_name == "mobilenetv2":
        backbone = mobilenet_v2(pretrained=True, progress=True)
        backbone.channels = [16, 24, 32, 96, 320]
    elif backbone_name == "convnextv2_nano":
        timm_backbone = timm.create_model(
            "convnextv2_nano", pretrained=False, features_only=True
        )
        backbone = TimmFeatureBackbone(timm_backbone, [80, 160, 320, 640])
        if not backbone_weight:
            raise ValueError("convnextv2_nano requires --backbone_weight")
        _load_local_backbone_weights(backbone.model, backbone_weight, backbone_name)
    else:
        raise NotImplementedError(
            "BACKBONE [%s] is not implemented! Supported backbones: mobilenetv2, convnextv2_nano\n"
            % backbone_name
        )
    return backbone


class PyramidFeatureFusion(nn.Module):
    def __init__(
        self,
        in_dims=[128, 128, 128, 128],
        dense_dim=1024,
        patch_size=16,
        hidden_dim=256,
    ):
        super().__init__()
        self.in_dims = in_dims
        self.dense_dim = dense_dim
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size

        self.c4 = nn.Sequential(
            DsBnRelu(in_dims[3] + hidden_dim, in_dims[3]), CBAM(in_dims[3], 8)
        )
        self.c3 = nn.Sequential(
            DsBnRelu(in_dims[2] + hidden_dim, in_dims[2]), CBAM(in_dims[2], 8)
        )
        self.c2 = nn.Sequential(
            DsBnRelu(in_dims[1] + hidden_dim, in_dims[1]), CBAM(in_dims[1], 8)
        )
        self.c1 = nn.Sequential(
            DsBnRelu(in_dims[0] + hidden_dim, in_dims[0]), CBAM(in_dims[0], 8)
        )

    def forward(self, feas, ds_feas):
        # process backbone (CNN) features
        x1, x2, x3, x4 = (
            feas  # [B, 128, 64, 64], [B, 128, 32, 32], [B, 128, 16, 16], [B, 128, 8, 8]
        )
        a1, a2, a3, a4 = (
            ds_feas  # [B, 256, 64, 64], [B, 256, 32, 32], [B, 256, 16, 16], [B, 256, 8, 8]
        )

        x4 = torch.cat([x4, a4], 1)
        x4 = self.c4(x4)

        x3 = torch.cat([x3, a3], 1)
        x3 = self.c3(x3)

        x2 = torch.cat([x2, a2], 1)
        x2 = self.c2(x2)

        x1 = torch.cat([x1, a1], 1)
        x1 = self.c1(x1)

        return x1, x2, x3, x4


class Encoder(nn.Module):
    def __init__(
        self,
        backbone="convnextv2_nano",
        fpn_channels=128,
        deform_groups=4,
        gamma_mode="SE",
        beta_mode="contextgatedconv",
        dino_arch="auto",
        dino_weight="dinov3/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
        backbone_weight=DEFAULT_CONVNEXTV2_NANO_WEIGHT,
        device="cuda",
        extract_ids=None,
        **kwargs,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.backbone = get_backbone(backbone, backbone_weight=backbone_weight)
        self.fpn = FPN(
            in_channels=self.backbone.channels[-4:],
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
            in_dims=[fpn_channels] * 4,
            dense_dim=self.dino.embed_dim,
            patch_size=self.dino.patch_size,
            hidden_dim=dense_out_dim,
        )

    def forward(self, x):
        """
        x : [B, 3, H, W]
        returns : (fea_pyramid, dino_deep)
            fea_pyramid : tuple of 4 FPN-DINO 融合特征
            dino_deep   : [B, D, H/8, W/8] DINOv3 最深层原始特征，用于
                          DinoContextBridge 的直达全局上下文注入，零额外计算开销
        """
        fea = self.backbone.forward(x)
        fea = self.fpn(fea[-4:])

        ds_fea_raw = self.dino(x)                # list of 4, each [B, D, H/8, W/8]

        ds_fea = self.dense_adp(ds_fea_raw)

        fea = self.pff(fea, ds_fea)

        return fea, ds_fea_raw[-1]


class FuseGated(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(nn.Conv2d(2 * dim, dim, 1, bias=True), nn.Sigmoid())
        self.mix = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        g = self.gate(torch.cat([x1, x2], dim=1))
        fused = x2 + g * x1
        return self.mix(fused)


class ContrastAwareDiff(nn.Module):
    """对比度感知差分模块。

    以 abs_diff 为基底（与 baseline 一致），通过轻量残差分支注入
    SAR 洪水检测的关键信号——有符号局部对比度变化 (Signed Local
    Contrast Delta)。对比度变化在洪水像素处为负（暗区被变亮邻域
    包围），在变亮建筑处为正，天然区分洪水与变亮误报。
    """

    def __init__(self, dim: int, pool_size: int = 5):
        super().__init__()
        self.local_pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size // 2
        )
        self.contrast_branch = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, aligned_pre, post):
        abs_diff = torch.abs(aligned_pre - post)

        pre_local = self.local_pool(aligned_pre)
        post_local = self.local_pool(post)
        delta_contrast = (post - post_local) - (aligned_pre - pre_local)

        contrast_feat = self.contrast_branch(delta_contrast)
        g = self.gate(contrast_feat)

        return abs_diff + g * contrast_feat


class DinoContextBridge(nn.Module):
    """DINOv3 全局上下文直注桥。

    P2 特征（64x64）交叉注意力查询 DINOv3 双时相全局 token，
    直接利用视觉基础模型已有的全局自注意力特征，
    替代 StripContextModule 从零重建全局上下文。

    复杂度 O(N x K)：N=4096 (P2 64x64), K=64 (DINO 8x8 池化)，
    远低于全局自注意力 O(N^2)=O(16M)。
    """

    def __init__(self, fpn_dim: int, dino_dim: int,
                 n_ctx_tokens: int = 64, num_heads: int = 4):
        super().__init__()
        G = int(n_ctx_tokens ** 0.5)     # 8x8 grid
        self.G = G
        self.num_heads = num_heads
        self.head_dim = fpn_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # DINOv3 双时相特征 → 全局变化语义投影
        self.dino_proj = nn.Sequential(
            nn.Conv2d(dino_dim * 2, fpn_dim, 1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.SiLU(inplace=True),
        )
        # P2 query 投影
        self.q_proj = nn.Conv2d(fpn_dim, fpn_dim, 1, bias=False)
        # DINO token → key/value
        self.kv_proj = nn.Linear(fpn_dim, fpn_dim * 2, bias=False)
        # 输出投影
        self.out_proj = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim, 1, bias=False),
            nn.BatchNorm2d(fpn_dim),
        )
        # 可学习门控，初始化为 0（训练初期不干扰已有特征）
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, p2_feat: torch.Tensor,
                dino_t1: torch.Tensor, dino_t2: torch.Tensor) -> torch.Tensor:
        """
        p2_feat : [B, C, H, W]    Detector P2 特征
        dino_t1 : [B, D, h, w]    Encoder 返回的 t1 DINOv3 最深层特征
        dino_t2 : [B, D, h, w]    Encoder 返回的 t2 DINOv3 最深层特征
        """
        B, C, H, W = p2_feat.shape

        # 拼接双时相 DINO 特征 → 投影为 FPN 维度
        dino_bi = torch.cat([dino_t1, dino_t2], dim=1)      # [B, 2D, h, w]
        dino_ctx = self.dino_proj(dino_bi)                   # [B, C, h, w]

        # 池化为少量全局 token
        tokens = F.adaptive_avg_pool2d(dino_ctx, (self.G, self.G))  # [B, C, G, G]
        tokens = tokens.flatten(2).permute(0, 2, 1)                  # [B, K, C]

        # P2 → query
        q = self.q_proj(p2_feat)                              # [B, C, H, W]
        q = q.flatten(2).permute(0, 2, 1)                    # [B, N, C]
        q = q.view(B, -1, self.num_heads, self.head_dim)     # [B, N, h, d]
        q = q.permute(0, 2, 1, 3)                            # [B, h, N, d]

        # DINO tokens → key, value
        kv = self.kv_proj(tokens)                             # [B, K, 2C]
        k, v = kv.chunk(2, dim=-1)
        k = k.view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Cross-attention: [B, h, N, d] x [B, h, K, d]^T → [B, h, N, K]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).permute(0, 2, 1, 3).contiguous()    # [B, N, h, d]
        out = out.view(B, H * W, C).permute(0, 2, 1).view(B, C, H, W)

        out = self.out_proj(out)
        return p2_feat + torch.tanh(self.gate) * out


class Detector(nn.Module):
    def __init__(
        self,
        fpn_channels=128,
        dino_embed_dim=384,
        n_layers=[1, 1, 1, 1],
        align_window=5,
        align_points=9,
        align_heads=4,
        align_on_levels=None,
        align_qkv_bias=False,
        align_offset_groups=4,
        contrast_pool_size=5,
        **kwargs,
    ):
        super().__init__()
        if align_on_levels is None:
            align_on_levels = [2, 3]
        self.align_on_levels = {int(level) for level in align_on_levels}

        self.soft_align_p2 = DeformableCrossAttentionAlign(
            dim=fpn_channels,
            num_heads=align_heads,
            num_points=align_points,
            window_size=align_window,
            offset_groups=align_offset_groups,
            qkv_bias=align_qkv_bias,
        )
        self.soft_align_p3 = DeformableCrossAttentionAlign(
            dim=fpn_channels,
            num_heads=align_heads,
            num_points=align_points,
            window_size=align_window,
            offset_groups=align_offset_groups,
            qkv_bias=align_qkv_bias,
        )
        self.diff_p2 = ContrastAwareDiff(fpn_channels, pool_size=contrast_pool_size)
        self.diff_p3 = ContrastAwareDiff(fpn_channels, pool_size=contrast_pool_size)
        self.diff_p4 = ContrastAwareDiff(fpn_channels, pool_size=contrast_pool_size)
        self.diff_p5 = ContrastAwareDiff(fpn_channels, pool_size=contrast_pool_size)
        self.p5_to_p4 = FuseGated(fpn_channels)
        self.p4_to_p3 = FuseGated(fpn_channels)
        self.p3_to_p2 = FuseGated(fpn_channels)

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
                    window_size=8,
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
        self.dino_ctx = DinoContextBridge(fpn_channels, dino_embed_dim)
        self.p5_head = nn.Conv2d(fpn_channels, 2, 1)
        self.p4_head = nn.Conv2d(fpn_channels, 2, 1)
        self.p3_head = nn.Conv2d(fpn_channels, 2, 1)
        self.p2_head = nn.Conv2d(fpn_channels, 2, 1)

    def forward(self, x1s, x2s, dino_t1, dino_t2, size=(256, 256)):
        """
        x1s, x2s  : Encoder 返回的 FPN 金字塔特征 (各 4 级)
        dino_t1   : [B, D, h, w] t1 DINOv3 最深层原始特征
        dino_t2   : [B, D, h, w] t2 DINOv3 最深层原始特征
        """
        t1_p2, t1_p3, t1_p4, t1_p5 = x1s
        t2_p2, t2_p3, t2_p4, t2_p5 = x2s

        aligned_pre_p2 = (
            self.soft_align_p2(t1_p2, t2_p2) if 2 in self.align_on_levels else t1_p2
        )
        aligned_pre_p3 = (
            self.soft_align_p3(t1_p3, t2_p3) if 3 in self.align_on_levels else t1_p3
        )
        aligned_pre_p4 = t1_p4
        aligned_pre_p5 = t1_p5

        diff_p2 = self.diff_p2(aligned_pre_p2, t2_p2)
        diff_p3 = self.diff_p3(aligned_pre_p3, t2_p3)
        diff_p4 = self.diff_p4(aligned_pre_p4, t2_p4)
        diff_p5 = self.diff_p5(aligned_pre_p5, t2_p5)

        fea_p5 = self.tb5(diff_p5)
        pred_p5 = self.p5_head(fea_p5)
        fea_p4 = self.p5_to_p4(fea_p5, diff_p4)
        fea_p4 = self.tb4(fea_p4)
        pred_p4 = self.p4_head(fea_p4)
        fea_p3 = self.p4_to_p3(fea_p4, diff_p3)
        fea_p3 = self.tb3(fea_p3)
        pred_p3 = self.p3_head(fea_p3)
        fea_p2 = self.p3_to_p2(fea_p3, diff_p2)
        fea_p2 = self.tb2(fea_p2)
        fea_p2 = self.dino_ctx(fea_p2, dino_t1, dino_t2)   # DINOv3 全局上下文直注
        pred_p2 = self.p2_head(fea_p2)

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

        return pred_p2, pred_p3, pred_p4, pred_p5, fea_p2


class ChangeModel(nn.Module):
    def __init__(
        self,
        backbone="convnextv2_nano",
        fpn_channels=128,
        n_layers=[1, 1, 1, 1],
        align_window=5,
        align_points=9,
        align_heads=4,
        align_on_levels=None,
        align_qkv_bias=False,
        align_offset_groups=4,
        contrast_pool_size=5,
        topo_grid_size=16,
        topo_hidden_dim=128,
        topo_neighbor_k=12,
        topo_n_hops=2,
        **kwargs,
    ):
        super().__init__()
        self.encoder = Encoder(backbone=backbone, fpn_channels=fpn_channels, **kwargs)
        self.detector = Detector(
            fpn_channels=fpn_channels,
            dino_embed_dim=self.encoder.dino.embed_dim,
            n_layers=n_layers,
            align_window=align_window,
            align_points=align_points,
            align_heads=align_heads,
            align_on_levels=align_on_levels,
            align_qkv_bias=align_qkv_bias,
            align_offset_groups=align_offset_groups,
            contrast_pool_size=contrast_pool_size,
            **kwargs,
        )
        self.refiner = FloodTopoRouter(
            feat_dim=fpn_channels,
            grid_size=topo_grid_size,
            hidden_dim=topo_hidden_dim,
            neighbor_k=topo_neighbor_k,
            n_hops=topo_n_hops,
        )

    @torch.inference_mode()
    def _forward(self, x1, x2):
        fea1, dino_deep1 = self.encoder(x1)
        fea2, dino_deep2 = self.encoder(x2)
        pred, _, _, _, det_p2_feat = self.detector(
            fea1, fea2, dino_deep1, dino_deep2, x1.shape[-2:]
        )
        pred, _ = self.refiner(pred, det_p2_feat)
        return pred

    def forward(self, x1, x2, gt_mask=None):
        fea1, dino_deep1 = self.encoder(x1)
        fea2, dino_deep2 = self.encoder(x2)

        pred_p2, pred_p3, pred_p4, pred_p5, det_p2_feat = self.detector(
            fea1, fea2, dino_deep1, dino_deep2
        )
        final_pred, topo_loss = self.refiner(
            pred_p2, det_p2_feat, gt_mask=gt_mask
        )
        return final_pred, (pred_p2, pred_p3, pred_p4, pred_p5), topo_loss
