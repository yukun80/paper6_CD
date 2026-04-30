import torch
import torch.nn.functional as F
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        if isinstance(alpha, (float, int)):
            alpha = [float(alpha), 1.0 - float(alpha)]
        self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))
        self.gamma = float(gamma)

    def forward(self, input, target):
        """计算二分类 focal loss；loss 内部固定使用 float32 保持 AMP 稳定。"""
        if input.ndim != 4 or input.shape[1] != 2:
            raise ValueError(f"FocalLoss expects logits with shape [B,2,H,W], got {tuple(input.shape)}")
        if target.ndim == 4 and target.shape[1] == 1:
            target = target.squeeze(1)

        input = input.float().permute(0, 2, 3, 1).reshape(-1, 2)
        target = target.long().reshape(-1)
        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = logpt.exp()

        alpha = self.alpha.to(device=input.device, dtype=input.dtype)
        alpha_t = alpha.gather(0, target)
        loss = -alpha_t * torch.pow(1.0 - pt, self.gamma) * logpt
        return loss.mean()


