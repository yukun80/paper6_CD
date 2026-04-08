"""对比度引导的双向精修器 (Contrast-Guided Bidirectional Refiner)

利用 Encoder 双时相特征的有符号局部对比度变化产生像素级校正图，
对 Detector 初步预测做双向调整：抑制变亮区域误报 + 增强暗区漏检。

核心物理依据：SAR 城市洪水场景中，灾后洪水像素相对局部邻域偏暗
（对比度为负），而变亮建筑像素相对局部邻域偏亮（对比度为正）。
该有符号对比度信号天然区分真实洪水与变亮假阳性。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastRefiner(nn.Module):
    """像素级双向对比度精修器，替换 FloodTopoRouter。

    Parameters
    ----------
    feat_dim : int
        Encoder p2 特征通道数 (= fpn_channels)。
    proj_dim : int
        特征投影维度，控制精修头的宽度。
    pool_k : int
        局部对比度估计的平均池化核大小，需为奇数。
    """

    def __init__(self, feat_dim: int, proj_dim: int = 32, pool_k: int = 7):
        super().__init__()
        self.pool_k = pool_k
        self.feat_proj = nn.Sequential(
            nn.Conv2d(feat_dim, proj_dim, 1, bias=False),
            nn.BatchNorm2d(proj_dim),
            nn.SiLU(inplace=True),
        )
        self.refine_head = nn.Sequential(
            nn.Conv2d(proj_dim + 1, proj_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(proj_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(proj_dim, 1, 1, bias=True),
        )
        self.alpha_raw = nn.Parameter(torch.tensor(-3.0))

    def forward(
        self,
        logit_2ch: torch.Tensor,
        fea_t1: torch.Tensor,
        fea_t2: torch.Tensor,
        gt_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Parameters
        ----------
        logit_2ch : [B, 2, H, W]  Detector 输出的初步 2 类 logits
        fea_t1    : [B, C, h, w]  Encoder T1 p2 特征
        fea_t2    : [B, C, h, w]  Encoder T2 p2 特征
        gt_mask   : 保留接口兼容，不使用

        Returns
        -------
        refined_logit : [B, 2, H, W]  精修后的 2 类 logits
        aux_loss      : None（无辅助损失）
        """
        B, _, H, W = logit_2ch.shape
        h, w = fea_t1.shape[-2:]

        p_fg = F.softmax(logit_2ch, dim=1)[:, 1:2]

        ft1 = self.feat_proj(fea_t1)
        ft2 = self.feat_proj(fea_t2)

        pad = self.pool_k // 2
        ft1_local = F.avg_pool2d(ft1, self.pool_k, 1, pad)
        ft2_local = F.avg_pool2d(ft2, self.pool_k, 1, pad)
        delta_c = (ft2 - ft2_local) - (ft1 - ft1_local)

        p_small = F.interpolate(
            p_fg, (h, w), mode="bilinear", align_corners=False
        )
        correction = torch.tanh(
            self.refine_head(torch.cat([delta_c, p_small], dim=1))
        )
        correction_up = F.interpolate(
            correction, (H, W), mode="bilinear", align_corners=False
        )

        alpha = torch.sigmoid(self.alpha_raw)
        corrected_p = (p_fg + alpha * correction_up).clamp(1e-6, 1 - 1e-6)
        corrected_logit = torch.logit(corrected_p, eps=1e-6)

        out = logit_2ch.clone()
        out[:, 1:2] = out[:, 1:2] + alpha * (corrected_logit - out[:, 1:2])

        return out, None
