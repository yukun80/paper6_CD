"""模型构建与优化器相关工具。"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    ResNet50_Weights,
    ResNet101_Weights,
    Swin_T_Weights,
    resnet50,
    resnet101,
    swin_t,
)
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.segmentation import fcn_resnet50, fcn_resnet101

try:
    from model.deeplabv3 import DeepLabHeadV3Plus, DeepLabV3
    from model.pspnet import PSPNet
    from model.unet import UNet
except ImportError:  # pragma: no cover
    from exp_template.model.deeplabv3 import DeepLabHeadV3Plus, DeepLabV3
    from exp_template.model.pspnet import PSPNet
    from exp_template.model.unet import UNet


def extract_logits_and_aux(output) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """兼容不同模型 forward 输出格式。"""

    if isinstance(output, dict):
        logits = output.get("out")
        aux = output.get("aux")
        if logits is None:
            raise ValueError("模型输出字典缺少 'out' 键")
        return logits, aux

    if isinstance(output, (tuple, list)):
        if len(output) == 0:
            raise ValueError("模型输出为空 tuple/list")
        logits = output[0]
        aux = output[1] if len(output) > 1 and torch.is_tensor(output[1]) else None
        return logits, aux

    if torch.is_tensor(output):
        return output, None

    raise TypeError(f"不支持的模型输出类型: {type(output)}")


def expand_input_conv(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """将预训练 3 通道首层卷积扩展到 N 通道。"""

    if conv.in_channels == in_channels:
        return conv

    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=(conv.bias is not None),
        padding_mode=conv.padding_mode,
    )

    with torch.no_grad():
        src_w = conv.weight.data
        old_in = src_w.shape[1]
        if in_channels > old_in:
            repeat = math.ceil(in_channels / old_in)
            new_w = src_w.repeat(1, repeat, 1, 1)[:, :in_channels, :, :]
            new_w = new_w * (old_in / float(in_channels))
        else:
            new_w = src_w[:, :in_channels, :, :]
        new_conv.weight.copy_(new_w)
        if conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(conv.bias.data)

    return new_conv


class SwinUPerLite(nn.Module):
    """Swin-T backbone + 轻量 FPN 解码头。"""

    def __init__(self, in_channels: int, num_classes: int, pretrained: bool = True, fpn_dim: int = 128) -> None:
        super().__init__()
        weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = swin_t(weights=weights)

        # patch embedding 首层通道扩展
        self.backbone.features[0][0] = expand_input_conv(self.backbone.features[0][0], in_channels)

        self.lateral1 = nn.Conv2d(96, fpn_dim, 1)
        self.lateral2 = nn.Conv2d(192, fpn_dim, 1)
        self.lateral3 = nn.Conv2d(384, fpn_dim, 1)
        self.lateral4 = nn.Conv2d(768, fpn_dim, 1)

        self.smooth1 = nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1)
        self.smooth2 = nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1)
        self.smooth3 = nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1)

        self.head = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_dim, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_size = x.shape[-2:]

        feats = self.backbone.features
        x = feats[0](x)
        x1 = feats[1](x)  # B,H/4,W/4,C96
        x2 = feats[2](x1)
        x2 = feats[3](x2)  # B,H/8,W/8,C192
        x3 = feats[4](x2)
        x3 = feats[5](x3)  # B,H/16,W/16,C384
        x4 = feats[6](x3)
        x4 = feats[7](x4)  # B,H/32,W/32,C768

        c1 = x1.permute(0, 3, 1, 2).contiguous()
        c2 = x2.permute(0, 3, 1, 2).contiguous()
        c3 = x3.permute(0, 3, 1, 2).contiguous()
        c4 = x4.permute(0, 3, 1, 2).contiguous()

        p4 = self.lateral4(c4)
        p3 = self.lateral3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.lateral2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        p1 = self.lateral1(c1) + F.interpolate(p2, size=c1.shape[-2:], mode="bilinear", align_corners=False)

        p3 = self.smooth3(p3)
        p2 = self.smooth2(p2)
        p1 = self.smooth1(p1)

        fuse = (
            p1
            + F.interpolate(p2, size=p1.shape[-2:], mode="bilinear", align_corners=False)
            + F.interpolate(p3, size=p1.shape[-2:], mode="bilinear", align_corners=False)
            + F.interpolate(p4, size=p1.shape[-2:], mode="bilinear", align_corners=False)
        ) / 4.0

        logits = self.head(fuse)
        return F.interpolate(logits, size=in_size, mode="bilinear", align_corners=False)


def _build_deeplabv3plus(model_cfg: Dict) -> nn.Module:
    in_channels = int(model_cfg.get("in_channels", 12))
    num_classes = int(model_cfg.get("num_classes", 3))
    backbone_name = str(model_cfg.get("backbone", "resnet50")).lower()
    pretrained = bool(model_cfg.get("pretrained", True))
    output_stride = int(model_cfg.get("output_stride", 16))

    if backbone_name not in {"resnet50", "resnet101"}:
        raise ValueError(f"DeepLabV3+ 仅支持 resnet50/resnet101，当前: {backbone_name}")

    if output_stride == 16:
        rsd = [False, False, True]
        aspp_dilate = [6, 12, 18]
    elif output_stride == 8:
        rsd = [False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        raise ValueError("output_stride 仅支持 8 或 16")

    if backbone_name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet50(weights=weights, replace_stride_with_dilation=rsd)
    else:
        weights = ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet101(weights=weights, replace_stride_with_dilation=rsd)

    backbone.conv1 = expand_input_conv(backbone.conv1, in_channels)
    return_layers = {"layer4": "out", "layer1": "low_level"}
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    classifier = DeepLabHeadV3Plus(
        in_channels=2048,
        low_level_channels=256,
        num_classes=num_classes,
        aspp_dilate=aspp_dilate,
    )
    model = DeepLabV3(backbone=backbone, classifier=classifier)
    return model


def _build_fcn_resnet(model_cfg: Dict) -> nn.Module:
    in_channels = int(model_cfg.get("in_channels", 12))
    num_classes = int(model_cfg.get("num_classes", 3))
    backbone = str(model_cfg.get("backbone", "resnet50")).lower()
    pretrained = bool(model_cfg.get("pretrained", True))

    if backbone == "resnet50":
        model = fcn_resnet50(
            weights=None,
            weights_backbone=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None,
            num_classes=num_classes,
        )
    elif backbone == "resnet101":
        model = fcn_resnet101(
            weights=None,
            weights_backbone=ResNet101_Weights.IMAGENET1K_V1 if pretrained else None,
            num_classes=num_classes,
        )
    else:
        raise ValueError(f"FCN-ResNet 仅支持 resnet50/resnet101，当前: {backbone}")

    model.backbone.conv1 = expand_input_conv(model.backbone.conv1, in_channels)
    return model


def _build_unet(model_cfg: Dict) -> nn.Module:
    return UNet(
        n_channels=int(model_cfg.get("in_channels", 12)),
        n_classes=int(model_cfg.get("num_classes", 3)),
        bilinear=bool(model_cfg.get("bilinear", False)),
    )


def _build_pspnet(model_cfg: Dict) -> nn.Module:
    model = PSPNet(
        layers=int(model_cfg.get("layers", 50)),
        bins=tuple(model_cfg.get("bins", [1, 2, 3, 6])),
        dropout=float(model_cfg.get("dropout", 0.1)),
        classes=int(model_cfg.get("num_classes", 3)),
        zoom_factor=int(model_cfg.get("zoom_factor", 1)),
        use_ppm=bool(model_cfg.get("use_ppm", True)),
        pretrained=bool(model_cfg.get("pretrained", False)),
        use_aux=bool(model_cfg.get("use_aux", True)),
    )

    in_channels = int(model_cfg.get("in_channels", 12))
    model.layer0[0] = expand_input_conv(model.layer0[0], in_channels)
    return model


def _build_swin_uperlite(model_cfg: Dict) -> nn.Module:
    return SwinUPerLite(
        in_channels=int(model_cfg.get("in_channels", 12)),
        num_classes=int(model_cfg.get("num_classes", 3)),
        pretrained=bool(model_cfg.get("pretrained", True)),
        fpn_dim=int(model_cfg.get("fpn_dim", 128)),
    )


def build_model(model_cfg: Dict) -> nn.Module:
    """根据 model.name 构建模型。"""

    name = str(model_cfg.get("name", "")).lower()
    if name in {"resnet", "fcn_resnet", "resnet_fcn"}:
        return _build_fcn_resnet(model_cfg)
    if name in {"unet"}:
        return _build_unet(model_cfg)
    if name in {"deeplabv3plus", "deeplabv3+", "deeplab"}:
        return _build_deeplabv3plus(model_cfg)
    if name in {"pspnet", "psp"}:
        return _build_pspnet(model_cfg)
    if name in {"swin", "swin_uperlite", "swin_transformer"}:
        return _build_swin_uperlite(model_cfg)
    raise ValueError(f"未知模型名: {name}")


def build_optimizer(model: nn.Module, optim_cfg: Dict) -> torch.optim.Optimizer:
    """按配置构建优化器。"""

    lr = float(optim_cfg.get("lr", 1e-4))
    weight_decay = float(optim_cfg.get("weight_decay", 1e-4))
    name = str(optim_cfg.get("type", "adamw")).lower()

    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        momentum = float(optim_cfg.get("momentum", 0.9))
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    raise ValueError(f"不支持的优化器: {name}")


def build_scheduler(optimizer: torch.optim.Optimizer, sched_cfg: Dict, max_epochs: int):
    """按配置构建学习率调度器。"""

    name = str(sched_cfg.get("type", "cosine")).lower()
    if name in {"none", ""}:
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_epochs,
            eta_min=float(sched_cfg.get("min_lr", 1e-6)),
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(sched_cfg.get("step_size", 20)),
            gamma=float(sched_cfg.get("gamma", 0.1)),
        )
    if name == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(sched_cfg.get("milestones", [30, 60])),
            gamma=float(sched_cfg.get("gamma", 0.1)),
        )
    raise ValueError(f"不支持的学习率调度器: {name}")


def build_criterion(loss_cfg: Dict, ignore_index: int) -> nn.Module:
    """构建主损失函数。"""

    class_weight = loss_cfg.get("class_weight", None)
    if class_weight is not None:
        weight = torch.tensor(class_weight, dtype=torch.float32)
    else:
        weight = None
    return nn.CrossEntropyLoss(ignore_index=ignore_index, weight=weight)
