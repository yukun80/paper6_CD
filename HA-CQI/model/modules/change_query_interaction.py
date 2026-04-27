import torch
import torch.nn as nn

from ..necks import DsBnRelu
from .attention_blocks import CBAM


class ChangeQueryInteractionBlock(nn.Module):
    """单尺度 CQI block：pair tokens 与 learnable change queries 双向交互。"""

    def __init__(self, channels: int, num_queries: int = 16, num_heads: int = 4, ffn_ratio: int = 2):
        super().__init__()
        self.change_queries = nn.Parameter(torch.randn(num_queries, channels) * 0.02)
        self.pair_projection = nn.Sequential(
            nn.Conv2d(channels * 4, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            DsBnRelu(channels, channels),
        )
        self.query_from_token = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.token_from_query = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.query_norm = nn.LayerNorm(channels)
        self.token_norm = nn.LayerNorm(channels)
        hidden_dim = channels * ffn_ratio
        self.ffn = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, channels),
        )
        self.out_projection = nn.Sequential(DsBnRelu(channels, channels), CBAM(channels, 8))

    def forward(self, pre_feat: torch.Tensor, post_feat: torch.Tensor) -> torch.Tensor:
        pair_feat = self.pair_projection(
            torch.cat([pre_feat, post_feat, post_feat - pre_feat, torch.abs(post_feat - pre_feat)], dim=1)
        )
        batch_size, channels, height, width = pair_feat.shape
        tokens = pair_feat.flatten(2).transpose(1, 2)
        queries = self.change_queries.unsqueeze(0).expand(batch_size, -1, -1)

        query_delta, _ = self.query_from_token(queries, tokens, tokens, need_weights=False)
        queries = self.query_norm(queries + query_delta)
        token_delta, _ = self.token_from_query(tokens, queries, queries, need_weights=False)
        tokens = self.token_norm(tokens + token_delta)
        tokens = self.token_norm(tokens + self.ffn(tokens))

        dense = tokens.transpose(1, 2).reshape(batch_size, channels, height, width)
        return self.out_projection(dense + pair_feat)


class ChangeQueryInteraction(nn.Module):
    """Module III：五尺度 CQI 生成结构化变化原语 D1-D5。"""

    def __init__(self, channels: int, num_queries: int = 16, num_heads: int = 4):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ChangeQueryInteractionBlock(
                    channels,
                    num_queries=num_queries,
                    num_heads=num_heads,
                )
                for _ in range(5)
            ]
        )

    def forward(self, pre_pyramid, post_pyramid):
        return tuple(
            block(pre_feat, post_feat)
            for block, pre_feat, post_feat in zip(self.blocks, pre_pyramid, post_pyramid)
        )
