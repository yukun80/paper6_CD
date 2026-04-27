import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbones import DEFAULT_BACKBONE_WEIGHT, build_feature_backbone
from ..necks import DsBnRelu, FPN
from .attention_blocks import CBAM
from .dino_adapter import DinoPyramidAdapter, DinoV3FeatureExtractor


class DinoSemanticFusion(nn.Module):
    """在 P3-P5 上用 DINOv3 语义锚校准 CNN 金字塔特征。"""

    def __init__(self, in_dims: list[int] | None = None, hidden_dim: int = 256):
        super().__init__()
        in_dims = in_dims or [128, 128, 128]
        self.blocks = nn.ModuleList(
            [nn.Sequential(DsBnRelu(dim + hidden_dim, dim), CBAM(dim, 8)) for dim in in_dims]
        )

    def forward(self, cnn_features, dino_features):
        if len(cnn_features) != len(dino_features) or len(cnn_features) != len(self.blocks):
            raise ValueError(
                f"DinoSemanticFusion expects matched lengths, got "
                f"{len(cnn_features)} CNN / {len(dino_features)} DINO / {len(self.blocks)} blocks"
            )
        return tuple(
            block(torch.cat([feat, dino_feat], dim=1))
            for feat, dino_feat, block in zip(cnn_features, dino_features, self.blocks)
        )


class HierarchicalCnnDinoEncoder(nn.Module):
    """Module I：共享双路 CNN-FPN-DINO 层次语义编码器。"""

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        fpn_channels: int = 128,
        deform_groups: int = 4,
        gamma_mode: str = "SE",
        beta_mode: str = "contextgatedconv",
        dino_arch: str = "auto",
        dino_weight: str = "dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        backbone_weight: str = DEFAULT_BACKBONE_WEIGHT,
        device: str = "cuda",
        extract_ids: list[int] | None = None,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        self.backbone_name = backbone
        self.backbone = build_feature_backbone(backbone, backbone_weight=backbone_weight)
        self.has_native_p1 = len(self.backbone.channels) == 5
        self.neck = FPN(
            in_channels=self.backbone.channels if self.has_native_p1 else self.backbone.channels[-4:],
            out_channels=fpn_channels,
            deform_groups=deform_groups,
            gamma_mode=gamma_mode,
            beta_mode=beta_mode,
        )
        dense_out_dim = fpn_channels * 2
        self.dino_extractor = DinoV3FeatureExtractor(
            dino_arch=dino_arch, weights_path=dino_weight, device=device, extract_ids=extract_ids
        )
        self.dino_adapter = DinoPyramidAdapter(
            in_dim=self.dino_extractor.embed_dim,
            out_dim=dense_out_dim,
            bottleneck=fpn_channels // 2,
        )
        self.semantic_fusion = DinoSemanticFusion(in_dims=[fpn_channels] * 3, hidden_dim=dense_out_dim)
        self.p1_from_p2 = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, 1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        cnn_stages = self.backbone.forward(x)
        pyramid = self.neck(cnn_stages if self.has_native_p1 else cnn_stages[-4:])

        dino_raw = self.dino_extractor(x)
        dino_features = self.dino_adapter(dino_raw[1:])

        if len(pyramid) == 5:
            p1, p2, p3, p4, p5 = pyramid
        else:
            p1 = self.p1_from_p2(
                F.interpolate(pyramid[0], scale_factor=2, mode="bilinear", align_corners=False)
            )
            p2, p3, p4, p5 = pyramid

        p3, p4, p5 = self.semantic_fusion((p3, p4, p5), dino_features)
        return p1, p2, p3, p4, p5
