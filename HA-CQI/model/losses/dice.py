import torch
import torch.nn.functional as F
import torch.nn as nn


class TverskyLoss(nn.Module):
    """前景优先 Tversky loss，用更高 FN 权重保护小目标召回。"""

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, eps: float = 1e-6):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = float(eps)

    def forward(self, input, target, alpha: float | None = None, beta: float | None = None):
        if input.ndim != 4 or input.shape[1] != 2:
            raise ValueError(f"TverskyLoss expects logits with shape [B,2,H,W], got {tuple(input.shape)}")
        if target.ndim == 4 and target.shape[1] == 1:
            target = target.squeeze(1)
        alpha = self.alpha if alpha is None else float(alpha)
        beta = self.beta if beta is None else float(beta)

        probs = F.softmax(input.float(), dim=1)[:, 1]
        target = target.long()
        target_fg = (target == 1).to(dtype=probs.dtype, device=probs.device)
        dims = tuple(range(1, probs.ndim))

        tp = torch.sum(probs * target_fg, dim=dims)
        fp = torch.sum(probs * (1.0 - target_fg), dim=dims)
        fn = torch.sum((1.0 - probs) * target_fg, dim=dims)
        score = (tp + self.eps) / (tp + alpha * fp + beta * fn + self.eps)
        return 1.0 - score.mean()


class DICELoss(TverskyLoss):
    """兼容旧导入名；当前实现为前景优先 Tversky loss。"""
