"""监督参考 prompt proposal 模型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def _load_resnet(name: str, pretrained: bool) -> nn.Module:
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        factory = models.resnet18
    elif name == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        factory = models.resnet34
    else:
        raise ValueError(f"Unsupported encoder: {name}")
    try:
        return factory(weights=weights)
    except Exception:
        return factory(weights=None)


def _binary_dilate_tensor(mask: torch.Tensor, ksize: int) -> torch.Tensor:
    pad = int(ksize // 2)
    return F.max_pool2d(mask, kernel_size=ksize, stride=1, padding=pad)


def _binary_erode_tensor(mask: torch.Tensor, ksize: int) -> torch.Tensor:
    pad = int(ksize // 2)
    return 1.0 - F.max_pool2d(1.0 - mask, kernel_size=ksize, stride=1, padding=pad)


class ResNetFPNEncoder(nn.Module):
    """共享的多尺度编码器。"""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        encoder_name: str = "resnet18",
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        backbone = _load_resnet(encoder_name, pretrained=pretrained)
        if in_channels != 3:
            old_conv = backbone.conv1
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
            with torch.no_grad():
                base = old_conv.weight.mean(dim=1, keepdim=True)
                new_conv.weight.copy_(base.repeat(1, in_channels, 1, 1) / max(in_channels / 3.0, 1.0))
            backbone.conv1 = new_conv
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        channels = [64, 128, 256, 512]
        self.lat1 = nn.Conv2d(channels[0], embed_dim, kernel_size=1)
        self.lat2 = nn.Conv2d(channels[1], embed_dim, kernel_size=1)
        self.lat3 = nn.Conv2d(channels[2], embed_dim, kernel_size=1)
        self.lat4 = nn.Conv2d(channels[3], embed_dim, kernel_size=1)
        self.smooth = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        feats = [
            F.interpolate(self.lat1(c1), size=input_size, mode="bilinear", align_corners=False),
            F.interpolate(self.lat2(c2), size=input_size, mode="bilinear", align_corners=False),
            F.interpolate(self.lat3(c3), size=input_size, mode="bilinear", align_corners=False),
            F.interpolate(self.lat4(c4), size=input_size, mode="bilinear", align_corners=False),
        ]
        return self.smooth(torch.cat(feats, dim=1))


@dataclass
class ProposalOutput:
    """参考引导 proposal 输出。"""

    positive_logit: torch.Tensor
    negative_logit: torch.Tensor
    segmentation_logit: torch.Tensor
    boundary_logit: torch.Tensor


class ReferencePromptProposalNet(nn.Module):
    """用多尺度 reference prototype 引导 query prompt 热图预测。"""

    def __init__(
        self,
        in_channels: int = 10,
        embed_dim: int = 64,
        hidden_dim: int = 96,
        encoder_type: str = "resnet_fpn",
        backbone_name: str = "resnet18",
        pretrained: bool = False,
        num_pos_prototypes: int = 3,
        num_neg_prototypes: int = 3,
        change_channel_idx: int = 7,
    ) -> None:
        super().__init__()
        if encoder_type != "resnet_fpn":
            raise ValueError("Only encoder_type='resnet_fpn' is supported in the upgraded pipeline")
        self.encoder = ResNetFPNEncoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            encoder_name=backbone_name,
            pretrained=pretrained,
        )
        self.in_channels = int(in_channels)
        self.change_channel_idx = int(change_channel_idx)
        self.num_pos_prototypes = int(num_pos_prototypes)
        self.num_neg_prototypes = int(num_neg_prototypes)
        fusion_in_channels = in_channels + embed_dim + 6 + 1
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_in_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 4, kernel_size=1),
        )

    def forward(
        self,
        query_inputs: torch.Tensor,
        reference_inputs: torch.Tensor,
        reference_mask: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> ProposalOutput:
        query_feat = self.encoder(query_inputs)
        ref_feat = self.encoder(reference_inputs)
        pos_sim_maps, neg_sim_maps = self._similarity_maps(query_feat, ref_feat, reference_mask, valid_mask)
        pos_max = torch.stack(pos_sim_maps, dim=1).max(dim=1, keepdim=True).values
        pos_mean = torch.stack(pos_sim_maps, dim=1).mean(dim=1, keepdim=True)
        neg_max = torch.stack(neg_sim_maps, dim=1).max(dim=1, keepdim=True).values
        neg_mean = torch.stack(neg_sim_maps, dim=1).mean(dim=1, keepdim=True)
        sim_gap = pos_max - neg_max
        sim_balance = pos_mean - neg_mean
        coarse_prior = query_inputs[:, self.change_channel_idx : self.change_channel_idx + 1]
        fused = torch.cat(
            [
                query_inputs,
                query_feat,
                pos_max,
                pos_mean,
                neg_max,
                neg_mean,
                sim_gap,
                sim_balance,
                coarse_prior,
            ],
            dim=1,
        )
        logits = self.fusion(fused)
        return ProposalOutput(
            positive_logit=logits[:, 0:1],
            negative_logit=logits[:, 1:2],
            segmentation_logit=logits[:, 2:3],
            boundary_logit=logits[:, 3:4],
        )

    def _similarity_maps(
        self,
        query_feat: torch.Tensor,
        ref_feat: torch.Tensor,
        reference_mask: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        flood_mask = (reference_mask > 0).float().unsqueeze(1)
        bg_mask = (reference_mask == 0).float().unsqueeze(1)
        if valid_mask is not None:
            valid = valid_mask.float().unsqueeze(1)
            flood_mask = flood_mask * valid
            bg_mask = bg_mask * valid
        flood_core = _binary_erode_tensor(flood_mask, 7)
        flood_boundary = torch.clamp(flood_mask - _binary_erode_tensor(flood_mask, 3), min=0.0)
        flood_boundary = torch.where(flood_boundary.sum(dim=(2, 3), keepdim=True) > 0, flood_boundary, flood_mask)
        flood_core = torch.where(flood_core.sum(dim=(2, 3), keepdim=True) > 0, flood_core, flood_mask)
        bg_ring = torch.clamp(_binary_dilate_tensor(flood_mask, 11) - flood_mask, min=0.0) * bg_mask
        bg_far = bg_mask * (1.0 - _binary_dilate_tensor(flood_mask, 19))
        bg_far = torch.where(bg_far.sum(dim=(2, 3), keepdim=True) > 0, bg_far, bg_mask)
        pos_masks = [flood_mask, flood_core, flood_boundary][: self.num_pos_prototypes]
        neg_masks = [bg_mask, bg_ring, bg_far][: self.num_neg_prototypes]
        qn = F.normalize(query_feat, dim=1)
        pos_sim_maps = [self._similarity_map(qn, ref_feat, mask) for mask in pos_masks]
        neg_sim_maps = [self._similarity_map(qn, ref_feat, mask) for mask in neg_masks]
        return pos_sim_maps, neg_sim_maps

    def _similarity_map(self, query_feat: torch.Tensor, ref_feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        proto = self._masked_average(ref_feat, mask)
        proto = F.normalize(proto, dim=1)
        return (query_feat * proto).sum(dim=1)

    def _masked_average(self, feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        denom = mask.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
        return (feat * mask).sum(dim=(2, 3), keepdim=True) / denom


def proposal_loss(output: ProposalOutput, gt: torch.Tensor, ignore_index: int = 255) -> Dict[str, torch.Tensor]:
    """参考引导 proposal 的监督损失。"""
    gt = gt.long()
    valid = gt != int(ignore_index)
    target = (gt > 0).float()
    if valid.sum() == 0:
        zero = output.segmentation_logit.sum() * 0.0
        return {"loss": zero, "bce": zero, "dice": zero, "focal": zero}

    seg_logit = output.segmentation_logit.squeeze(1)
    boundary_target = _boundary_target(target)
    pos_count = target[valid].sum().clamp_min(1.0)
    neg_count = (1.0 - target[valid]).sum().clamp_min(1.0)
    pos_weight = (neg_count / pos_count).clamp(1.0, 12.0)
    bce = F.binary_cross_entropy_with_logits(seg_logit[valid], target[valid], pos_weight=pos_weight)
    prob = torch.sigmoid(seg_logit)
    inter = (prob[valid] * target[valid]).sum()
    false_pos = (prob[valid] * (1.0 - target[valid])).sum()
    false_neg = ((1.0 - prob[valid]) * target[valid]).sum()
    dice = 1.0 - (inter + 1.0) / (inter + 0.35 * false_pos + 0.65 * false_neg + 1.0)
    focal = _binary_focal_loss(seg_logit[valid], target[valid])
    aux_pos = F.binary_cross_entropy_with_logits(output.positive_logit.squeeze(1)[valid], target[valid], pos_weight=pos_weight)
    aux_neg = F.binary_cross_entropy_with_logits(output.negative_logit.squeeze(1)[valid], 1.0 - target[valid])
    boundary = F.binary_cross_entropy_with_logits(output.boundary_logit.squeeze(1)[valid], boundary_target[valid])
    loss = 0.65 * bce + 0.85 * dice + 0.35 * focal + 0.20 * (aux_pos + aux_neg) + 0.15 * boundary
    return {"loss": loss, "bce": bce.detach(), "dice": dice.detach(), "focal": focal.detach()}


def _binary_focal_loss(logit: torch.Tensor, target: torch.Tensor, alpha: float = 0.75, gamma: float = 2.0) -> torch.Tensor:
    prob = torch.sigmoid(logit)
    pt = torch.where(target > 0.5, prob, 1.0 - prob)
    alpha_t = torch.where(target > 0.5, torch.full_like(target, alpha), torch.full_like(target, 1.0 - alpha))
    loss = -alpha_t * ((1.0 - pt).clamp_min(1e-6) ** gamma) * torch.log(pt.clamp_min(1e-6))
    return loss.mean()


def _boundary_target(target: torch.Tensor) -> torch.Tensor:
    target = target.unsqueeze(1)
    dilated = _binary_dilate_tensor(target, 3)
    eroded = _binary_erode_tensor(target, 3)
    return torch.clamp(dilated - eroded, min=0.0).squeeze(1)
