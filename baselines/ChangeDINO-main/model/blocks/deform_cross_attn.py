import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeformableCrossAttentionAlign(nn.Module):
    """基于灾后查询的局部可变形交叉注意力，对灾前特征做软对齐。"""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        num_points: int = 9,
        window_size: int = 5,
        offset_groups: int = 4,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        if window_size < 3 or window_size % 2 == 0:
            raise ValueError(f"window_size must be odd and >= 3, got {window_size}")
        if num_points < 1:
            raise ValueError(f"num_points must be positive, got {num_points}")
        if offset_groups < 1 or (dim * 3) % offset_groups != 0 or dim % offset_groups != 0:
            raise ValueError(
                "offset_groups must divide both 3*dim and dim for grouped offset prediction"
            )

        self.dim = dim
        self.num_heads = num_heads
        self.num_points = num_points
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.offset_scale = max(1.0, window_size // 2 / 2)

        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.k_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

        # 用 pre/post/delta 联合预测局部偏移和注意力先验。
        self.offset_predictor = nn.Sequential(
            nn.Conv2d(dim * 3, dim, kernel_size=3, padding=1, groups=offset_groups, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim, num_heads * num_points * 3, kernel_size=1, bias=True),
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )

        self.register_buffer(
            "reference_points",
            self._build_reference_points(window_size=window_size, num_points=num_points),
            persistent=False,
        )
        self.register_buffer(
            "dot_scale", torch.tensor(self.head_dim**-0.5), persistent=False
        )

    @staticmethod
    def _build_reference_points(window_size: int, num_points: int) -> torch.Tensor:
        """在局部窗口内生成固定参考采样点，偏移分支只学习细调量。"""
        radius = window_size // 2
        side = int(math.sqrt(num_points))
        if side * side == num_points:
            anchors = torch.linspace(-radius, radius, steps=side)
            yy, xx = torch.meshgrid(anchors, anchors, indexing="ij")
            coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
            return coords

        yy, xx = torch.meshgrid(
            torch.arange(-radius, radius + 1, dtype=torch.float32),
            torch.arange(-radius, radius + 1, dtype=torch.float32),
            indexing="ij",
        )
        coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
        if num_points <= coords.shape[0]:
            return coords[:num_points]
        repeat = math.ceil(num_points / coords.shape[0])
        return coords.repeat(repeat, 1)[:num_points]

    @staticmethod
    def _make_base_grid(height: int, width: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        ys = torch.arange(height, device=device, dtype=dtype)
        xs = torch.arange(width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        return grid_x, grid_y

    @staticmethod
    def _normalize_grid(sample_x: torch.Tensor, sample_y: torch.Tensor, height: int, width: int) -> torch.Tensor:
        denom_x = max(width - 1, 1)
        denom_y = max(height - 1, 1)
        x_norm = 2.0 * sample_x / denom_x - 1.0
        y_norm = 2.0 * sample_y / denom_y - 1.0
        return torch.stack([x_norm, y_norm], dim=-1)

    def forward(self, pre_feat: torch.Tensor, post_feat: torch.Tensor) -> torch.Tensor:
        """以 post 作为 query，在 pre 局部邻域内检索最匹配语义，返回对齐后的 pre。"""
        if pre_feat.shape != post_feat.shape:
            raise ValueError(
                f"pre/post feature shapes must match, got {pre_feat.shape} vs {post_feat.shape}"
            )

        batch_size, _, height, width = pre_feat.shape
        query = self.q_proj(post_feat).view(
            batch_size, self.num_heads, self.head_dim, height, width
        )
        key_map = self.k_proj(pre_feat).view(
            batch_size, self.num_heads, self.head_dim, height, width
        )
        value_map = self.v_proj(pre_feat).view(
            batch_size, self.num_heads, self.head_dim, height, width
        )

        delta = post_feat - pre_feat
        offset_logits = self.offset_predictor(torch.cat([pre_feat, post_feat, delta], dim=1))
        offset_logits = offset_logits.view(
            batch_size, self.num_heads, self.num_points, 3, height, width
        )

        offsets = torch.tanh(offset_logits[:, :, :, :2]) * self.offset_scale
        attn_bias = offset_logits[:, :, :, 2].permute(0, 1, 3, 4, 2).contiguous()

        base_x, base_y = self._make_base_grid(height, width, pre_feat.device, pre_feat.dtype)
        base_x = base_x.view(1, 1, 1, height, width)
        base_y = base_y.view(1, 1, 1, height, width)
        ref_points = self.reference_points.to(device=pre_feat.device, dtype=pre_feat.dtype)
        ref_x = ref_points[:, 0].view(1, 1, self.num_points, 1, 1)
        ref_y = ref_points[:, 1].view(1, 1, self.num_points, 1, 1)

        sample_x = base_x + ref_x + offsets[:, :, :, 0]
        sample_y = base_y + ref_y + offsets[:, :, :, 1]
        sample_grid = self._normalize_grid(sample_x, sample_y, height, width)
        sample_grid = sample_grid.permute(0, 1, 3, 4, 2, 5).contiguous()
        sample_grid = sample_grid.view(batch_size * self.num_heads, height, width * self.num_points, 2)

        key_samples = F.grid_sample(
            key_map.view(batch_size * self.num_heads, self.head_dim, height, width),
            sample_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        value_samples = F.grid_sample(
            value_map.view(batch_size * self.num_heads, self.head_dim, height, width),
            sample_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        key_samples = key_samples.view(
            batch_size, self.num_heads, self.head_dim, height, width, self.num_points
        )
        value_samples = value_samples.view(
            batch_size, self.num_heads, self.head_dim, height, width, self.num_points
        )

        scores = (
            (query.unsqueeze(-1) * key_samples).sum(dim=2) * self.dot_scale + attn_bias
        )
        attn = scores.softmax(dim=-1)
        aligned = (attn.unsqueeze(2) * value_samples).sum(dim=-1)
        aligned = aligned.view(batch_size, self.dim, height, width)

        # 保留原始灾前特征的稳定基底，学习局部校正残差。
        return pre_feat + self.out_proj(aligned)
