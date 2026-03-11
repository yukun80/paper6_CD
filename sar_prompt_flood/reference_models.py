"""监督参考 prompt proposal 模型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """轻量卷积块。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class ProposalOutput:
    """参考引导 proposal 输出。"""

    positive_logit: torch.Tensor
    negative_logit: torch.Tensor
    segmentation_logit: torch.Tensor


class ReferencePromptProposalNet(nn.Module):
    """用参考样本 prototype 引导 query prompt 热图预测。"""

    def __init__(self, in_channels: int = 8, embed_dim: int = 32, hidden_dim: int = 48) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, hidden_dim),
            ConvBlock(hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim, embed_dim, kernel_size=1),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels + 3, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 2, kernel_size=1),
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

        flood_mask = (reference_mask > 0).float()
        bg_mask = (reference_mask == 0).float()
        if valid_mask is not None:
            bg_mask = bg_mask * valid_mask.float()
            flood_mask = flood_mask * valid_mask.float()

        pos_proto = self._masked_average(ref_feat, flood_mask)
        neg_proto = self._masked_average(ref_feat, bg_mask)

        qn = F.normalize(query_feat, dim=1)
        pos_proto = F.normalize(pos_proto, dim=1)
        neg_proto = F.normalize(neg_proto, dim=1)
        sim_pos = (qn * pos_proto).sum(dim=1, keepdim=True)
        sim_neg = (qn * neg_proto).sum(dim=1, keepdim=True)
        sim_gap = sim_pos - sim_neg
        fused = torch.cat([query_inputs, sim_pos, sim_neg, sim_gap], dim=1)
        logits = self.fusion(fused)
        positive_logit = logits[:, :1]
        negative_logit = logits[:, 1:2]
        segmentation_logit = positive_logit - negative_logit
        return ProposalOutput(
            positive_logit=positive_logit,
            negative_logit=negative_logit,
            segmentation_logit=segmentation_logit,
        )

    def _masked_average(self, feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(1)
        denom = mask.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
        pooled = (feat * mask).sum(dim=(2, 3), keepdim=True) / denom
        return pooled


def proposal_loss(output: ProposalOutput, gt: torch.Tensor, ignore_index: int = 255) -> Dict[str, torch.Tensor]:
    """参考引导 proposal 的监督损失。"""
    gt = gt.long()
    valid = gt != int(ignore_index)
    target = (gt > 0).float()
    if valid.sum() == 0:
        zero = output.segmentation_logit.sum() * 0.0
        return {"loss": zero, "bce": zero, "dice": zero}

    seg_logit = output.segmentation_logit.squeeze(1)
    bce = F.binary_cross_entropy_with_logits(seg_logit[valid], target[valid])
    prob = torch.sigmoid(seg_logit)
    inter = (prob[valid] * target[valid]).sum()
    denom = prob[valid].sum() + target[valid].sum()
    dice = 1.0 - (2.0 * inter + 1.0) / (denom + 1.0)
    aux_pos = F.binary_cross_entropy_with_logits(output.positive_logit.squeeze(1)[valid], target[valid])
    aux_neg = F.binary_cross_entropy_with_logits(output.negative_logit.squeeze(1)[valid], 1.0 - target[valid])
    loss = bce + dice + 0.25 * (aux_pos + aux_neg)
    return {"loss": loss, "bce": bce.detach(), "dice": dice.detach()}
