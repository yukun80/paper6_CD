import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatioTemporalContrastGate(nn.Module):
    """城市内涝场景下，利用局部双域对比度的剧烈反转来验证初步检测
    结果中的真实洪水区域，门控抑制配准残差和过度推理产生的假阳性。

    三步核心机制
    ----------
    1. 形态自适应软边界分离 —— 用 Base Mask 动态划分内部积水区与外部护城河
    2. 局部不规则掩膜池化   —— 在 T1/T2 特征图上沿不规则掩膜做局部加权平均
    3. 时空对比度演变门控   —— 计算空间反差的时间变化量，逐像素门控抑制假阳性
    """

    def __init__(
        self,
        feat_dim: int,
        dilation_k: int = 5,
        pool_k: int = 11,
        tau: float = 0.05,
    ):
        super().__init__()
        self.dilation_k = dilation_k
        self.pool_k = pool_k

        # ── Step 1 参数：软边界分离 ──
        self.conf_logit = nn.Parameter(torch.tensor(0.0))
        n = dilation_k * dilation_k
        self.dilation_w = nn.Parameter(torch.zeros(1, n))
        self.log_tau = nn.Parameter(torch.log(torch.tensor(float(tau))))

        # ── Step 2 参数：特征投影 + 局部掩膜池化 ──
        proj_dim = max(feat_dim // 4, 32)
        self.proj_dim = proj_dim
        self.feat_proj = nn.Sequential(
            nn.Conv2d(feat_dim, proj_dim, 1, bias=False),
            nn.BatchNorm2d(proj_dim),
            nn.SiLU(inplace=True),
        )
        self.register_buffer(
            "_sum_kernel", torch.ones(proj_dim, 1, pool_k, pool_k)
        )
        self.register_buffer(
            "_count_kernel", torch.ones(1, 1, pool_k, pool_k)
        )

        # ── Step 3 参数：对比度演变 → 标量门控 ──
        self.contrast_to_gate = nn.Sequential(
            nn.Conv2d(proj_dim, proj_dim, 1, bias=False),
            nn.BatchNorm2d(proj_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(proj_dim, 1, 1, bias=True),
        )

        # 残差混合系数，初始化为较小值让训练初期 STCG 对输出扰动较小
        self.alpha_raw = nn.Parameter(torch.tensor(0.0))

    # ------------------------------------------------------------------ #
    #  Step 1 辅助：可微软膨胀 (logsumexp ≈ max-pooling)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _logsumexp_pool(x_cols: torch.Tensor, w: torch.Tensor, tau: float):
        """x_cols: [B,1,K,HW], w: [1,K], tau: scalar"""
        return torch.logsumexp((x_cols + w.unsqueeze(-1)) / tau, dim=2) * tau

    def _soft_dilate(self, x: torch.Tensor, k: int,
                     w: torch.Tensor, tau: float) -> torch.Tensor:
        if k <= 1:
            return x
        B, _, H, W = x.shape
        pad = k // 2
        cols = F.unfold(x, k, padding=pad).view(B, 1, k * k, H * W)
        z = self._logsumexp_pool(cols, w, tau)
        return z.view(B, 1, H, W)

    # ------------------------------------------------------------------ #
    #  Step 2 辅助：基于卷积的局部掩膜加权平均
    # ------------------------------------------------------------------ #

    def _local_masked_avg(self, feat: torch.Tensor,
                          mask: torch.Tensor) -> torch.Tensor:
        """在每个像素的 pool_k x pool_k 邻域内，按 mask 加权求特征均值。

        feat : [B, proj_dim, h, w]
        mask : [B, 1, h, w]  soft mask ∈ [0,1]
        return: [B, proj_dim, h, w]
        """
        pad = self.pool_k // 2
        masked_feat = feat * mask
        numerator = F.conv2d(
            masked_feat, self._sum_kernel, padding=pad, groups=self.proj_dim
        )
        denominator = F.conv2d(mask, self._count_kernel, padding=pad)
        return numerator / denominator.clamp_min(1e-6)

    # ------------------------------------------------------------------ #
    #  forward
    # ------------------------------------------------------------------ #

    def forward(
        self,
        logit_2ch: torch.Tensor,
        fea_t1: torch.Tensor,
        fea_t2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logit_2ch : [B, 2, H, W]  Detector 输出的初步 2 类 logits (Base Mask)
        fea_t1    : [B, C, h, w]  Encoder T1 p2 特征
        fea_t2    : [B, C, h, w]  Encoder T2 p2 特征

        Returns
        -------
        [B, 2, H, W]  门控校验后的 2 类 logits
        """
        B, C_logit, H, W = logit_2ch.shape
        h, w = fea_t1.shape[-2:]

        p_fg = F.softmax(logit_2ch, dim=1)[:, 1:2]          # [B,1,H,W]

        # ============ Step 1: 形态自适应的软边界分离 ============
        conf = torch.sigmoid(self.conf_logit)
        tau = torch.exp(self.log_tau).clamp_min(1e-4)

        # 内部积水区：steep sigmoid 近似硬阈值但保持可微
        inner_mask = torch.sigmoid((p_fg - conf) * 20.0)      # [B,1,H,W]

        # 软膨胀扩展边界
        dilated = self._soft_dilate(p_fg, self.dilation_k, self.dilation_w, tau)
        outer_boundary = torch.sigmoid((dilated - conf * 0.5) * 20.0)

        # 外部护城河 = 膨胀区 − 内部区
        outer_mask = (outer_boundary - inner_mask).clamp_min(0)

        # 下采样到 p2 特征分辨率
        inner_f = F.interpolate(inner_mask, size=(h, w),
                                mode="bilinear", align_corners=False)
        outer_f = F.interpolate(outer_mask, size=(h, w),
                                mode="bilinear", align_corners=False)

        # ============ Step 2: 局部不规则掩膜池化 ============
        ft1 = self.feat_proj(fea_t1)                          # [B, proj_dim, h, w]
        ft2 = self.feat_proj(fea_t2)

        inner_t1 = self._local_masked_avg(ft1, inner_f)
        inner_t2 = self._local_masked_avg(ft2, inner_f)
        outer_t1 = self._local_masked_avg(ft1, outer_f)
        outer_t2 = self._local_masked_avg(ft2, outer_f)

        # ============ Step 3: 时空对比度演变与门控 ============
        c_pre  = inner_t1 - outer_t1                          # 灾前空间对比度
        c_post = inner_t2 - outer_t2                          # 灾后空间对比度
        delta_c = (c_post - c_pre).abs()                      # 时空演变强度

        gate_logit = self.contrast_to_gate(delta_c)           # [B,1,h,w]
        gate = torch.sigmoid(gate_logit)

        # 上采样门控到 logits 全分辨率
        gate_up = F.interpolate(gate, size=(H, W),
                                mode="bilinear", align_corners=False)

        # ΔC 大 → gate≈1 → 保留；ΔC 小 → gate≈0 → 抑制
        gated_p = p_fg * gate_up
        gated_logit = torch.logit(gated_p.clamp(1e-6, 1 - 1e-6), eps=1e-6)

        alpha = torch.sigmoid(self.alpha_raw)
        out = logit_2ch.clone()
        out[:, 1:2] = out[:, 1:2] + alpha * (gated_logit - out[:, 1:2])

        return out