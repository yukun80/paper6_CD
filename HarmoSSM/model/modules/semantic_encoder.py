import torch
import torch.nn as nn

from ..backbones import (
    DEFAULT_BACKBONE_NAME,
    DEFAULT_BACKBONE_WEIGHT,
    build_feature_backbone,
)
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
        fpn_channels: int = 128,
        deform_groups: int = 4,
        gamma_mode: str = "SE",
        beta_mode: str = "contextgatedconv",
        dino_arch: str = "auto",
        dino_weight: str = "dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        backbone_weight: str = DEFAULT_BACKBONE_WEIGHT,
        device: str = "cuda",
        dino_fusion_layers: list[int] | None = None,
        input_mean: list[float] | None = None,
        input_std: list[float] | None = None,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        self.backbone_name = DEFAULT_BACKBONE_NAME
        input_mean = input_mean or [0.5, 0.5, 0.5]
        input_std = input_std or [0.5, 0.5, 0.5]
        if len(input_mean) != 3 or len(input_std) != 3:
            raise ValueError("input_mean/input_std must contain exactly three values")
        if any(float(value) <= 0.0 for value in input_std):
            raise ValueError("input_std values must be positive")
        self.register_buffer(
            "input_mean",
            torch.tensor(input_mean, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "input_std",
            torch.tensor(input_std, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "dino_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "dino_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.backbone = build_feature_backbone(backbone_weight=backbone_weight)
        self.neck = FPN(
            in_channels=self.backbone.channels,
            out_channels=fpn_channels,
            deform_groups=deform_groups,
            gamma_mode=gamma_mode,
            beta_mode=beta_mode,
        )
        dense_out_dim = fpn_channels * 2
        self.dino_extractor = DinoV3FeatureExtractor(
            dino_arch=dino_arch,
            weights_path=dino_weight,
            device=device,
            fusion_layers=dino_fusion_layers,
        )
        self.dino_adapter = DinoPyramidAdapter(
            in_dim=self.dino_extractor.embed_dim,
            out_dim=dense_out_dim,
            bottleneck=fpn_channels // 2,
        )
        self.semantic_fusion = DinoSemanticFusion(in_dims=[fpn_channels] * 3, hidden_dim=dense_out_dim)

    def prepare_dino_input(self, x: torch.Tensor) -> torch.Tensor:
        """把 CNN 的数据集归一化输入转换为 DINOv3-LVD 官方输入。"""
        raw = torch.clamp(x * self.input_std + self.input_mean, 0.0, 1.0)
        return (raw - self.dino_mean) / self.dino_std

    def forward(self, x):
        cnn_stages = self.backbone(x)
        pyramid = self.neck(cnn_stages)

        dino_raw = self.dino_extractor(self.prepare_dino_input(x))
        dino_features = self.dino_adapter(dino_raw)

        p1, p2, p3, p4, p5 = pyramid
        p3, p4, p5 = self.semantic_fusion((p3, p4, p5), dino_features)
        return p1, p2, p3, p4, p5
