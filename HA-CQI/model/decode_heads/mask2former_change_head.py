import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.attention_blocks import CBAM
from ..necks import DsBnRelu


class MultiScalePixelDecoder(nn.Module):
    """将多尺度变化原语融合为 dense mask feature。"""

    def __init__(self, channels: int, mask_dim: int):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(channels, mask_dim, 1, bias=False) for _ in range(5)])
        self.smooth = nn.ModuleList([DsBnRelu(mask_dim, mask_dim) for _ in range(5)])
        self.out = nn.Sequential(DsBnRelu(mask_dim, mask_dim), CBAM(mask_dim, 8))

    def forward(self, features):
        projected = [smooth(lateral(feat)) for feat, lateral, smooth in zip(features, self.lateral, self.smooth)]
        x = projected[-1]
        for feat in reversed(projected[:-1]):
            x = F.interpolate(x, size=feat.shape[-2:], mode="bilinear", align_corners=False)
            x = x + feat
        return self.out(x)


class MaskQueryDecoderLayer(nn.Module):
    """Mask query decoder 层，用多尺度变化原语更新 mask queries。"""

    def __init__(self, mask_dim: int, num_heads: int = 4, ffn_ratio: int = 2):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(mask_dim, num_heads, batch_first=True)
        self.self_attn = nn.MultiheadAttention(mask_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(mask_dim)
        self.norm2 = nn.LayerNorm(mask_dim)
        self.norm3 = nn.LayerNorm(mask_dim)
        hidden_dim = mask_dim * ffn_ratio
        self.ffn = nn.Sequential(nn.Linear(mask_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, mask_dim))

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        delta, _ = self.cross_attn(queries, memory, memory, need_weights=False)
        queries = self.norm1(queries + delta)
        delta, _ = self.self_attn(queries, queries, queries, need_weights=False)
        queries = self.norm2(queries + delta)
        return self.norm3(queries + self.ffn(queries))


class Mask2FormerChangeHead(nn.Module):
    """标准 Mask2Former-style head，用 CQI 原语预测二值洪水变化 mask。"""

    def __init__(
        self,
        channels: int,
        mask_dim: int = 128,
        num_mask_queries: int = 32,
        num_decoder_layers: int = 3,
        num_heads: int = 4,
    ):
        super().__init__()
        self.mask_queries = nn.Parameter(torch.randn(num_mask_queries, mask_dim) * 0.02)
        self.pixel_decoder = MultiScalePixelDecoder(channels, mask_dim)
        self.memory_proj = nn.ModuleList([nn.Conv2d(channels, mask_dim, 1, bias=False) for _ in range(5)])
        self.decoder_layers = nn.ModuleList(
            [MaskQueryDecoderLayer(mask_dim, num_heads=num_heads) for _ in range(num_decoder_layers)]
        )
        self.class_embed = nn.Linear(mask_dim, 2)
        self.mask_embed = nn.Sequential(
            nn.Linear(mask_dim, mask_dim),
            nn.GELU(),
            nn.Linear(mask_dim, mask_dim),
        )
        self.semantic_head = nn.Sequential(DsBnRelu(mask_dim, mask_dim), nn.Conv2d(mask_dim, 2, 1))
        self.query_alpha = nn.Parameter(torch.tensor(0.0))

    def _build_memory(self, features, target_size: tuple[int, int]) -> torch.Tensor:
        memories = []
        for feat, proj in zip(features, self.memory_proj):
            mem = proj(feat)
            if mem.shape[-2:] != target_size:
                mem = F.interpolate(mem, size=target_size, mode="bilinear", align_corners=False)
            memories.append(mem)
        memory = torch.stack(memories, dim=0).mean(dim=0)
        return memory.flatten(2).transpose(1, 2)

    def forward(self, features, output_size: tuple[int, int]) -> torch.Tensor:
        mask_feature = self.pixel_decoder(features)
        batch_size, _, height, width = mask_feature.shape
        memory = self._build_memory(features, target_size=(height, width))
        queries = self.mask_queries.unsqueeze(0).expand(batch_size, -1, -1)
        for layer in self.decoder_layers:
            queries = layer(queries, memory)

        class_logits = self.class_embed(queries)
        mask_kernels = self.mask_embed(queries)
        mask_logits = torch.einsum("bqc,bchw->bqhw", mask_kernels, mask_feature)
        class_prob = torch.softmax(class_logits, dim=-1)
        query_logits = torch.einsum("bqc,bqhw->bchw", class_prob, mask_logits)
        logits = self.semantic_head(mask_feature) + torch.tanh(self.query_alpha) * query_logits
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
