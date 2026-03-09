import torch
import torch.nn.functional as F

# Multi-Scale BCEDice Loss
# BCEDiceLoss1 + BCEDiceLoss2 + BCEDiceLoss3 + BCEDiceLoss4

def BCEDiceLoss(inputs, targets):
    # 使用 logits 版本 BCE，保证与 autocast/AMP 兼容。
    bce = F.binary_cross_entropy_with_logits(inputs, targets)
    probs = torch.sigmoid(inputs)
    inter = (probs * targets).sum()
    eps = 1e-5
    dice = (2 * inter + eps) / (probs.sum() + targets.sum() + eps)
    return bce + 1 - dice
