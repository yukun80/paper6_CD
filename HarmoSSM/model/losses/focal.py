"""二分类 Focal loss。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, class_weights, gamma: float = 2.0):
        super().__init__()
        weights = torch.as_tensor(class_weights, dtype=torch.float32)
        if weights.shape != (2,):
            raise ValueError(
                "FocalLoss expects class_weights=[BACKGROUND, FOREGROUND], "
                f"got shape {tuple(weights.shape)}"
            )
        if torch.any(weights < 0) or float(weights.sum()) <= 0.0:
            raise ValueError(
                "FocalLoss class weights must be non-negative with a positive sum"
            )
        weights = weights / weights.sum()
        self.register_buffer("class_weights", weights)
        self.gamma = float(gamma)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算二分类 focal loss；loss 内部固定使用 float32 保持 AMP 稳定。"""
        if input.ndim != 4 or input.shape[1] != 2:
            raise ValueError(
                "FocalLoss expects logits with shape [B,2,H,W], "
                f"got {tuple(input.shape)}"
            )
        if target.ndim == 4 and target.shape[1] == 1:
            target = target.squeeze(1)

        input = input.float().permute(0, 2, 3, 1).reshape(-1, 2)
        target = target.long().reshape(-1)
        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = logpt.exp()

        weights = self.class_weights.to(device=input.device, dtype=input.dtype)
        sample_weights = weights.gather(0, target)
        loss = -sample_weights * torch.pow(1.0 - pt, self.gamma) * logpt
        return loss.mean()
