"""OSCD 使用的四方向二维 selective scan。

CUDA 路径只调用 ``mamba_ssm`` 的预编译 selective-scan kernel；CPU 路径保留
一个可微的 FP32 数学参考实现，供单元测试和结构检查使用。CUDA 缺少 kernel
时显式失败，避免训练意外退化为极慢的 Python recurrence。
"""

from __future__ import annotations

import math
from importlib.metadata import PackageNotFoundError, version
from typing import Final

import torch
from torch import nn
import torch.nn.functional as F


EXPECTED_MAMBA_VERSION: Final[str] = "2.2.4"

try:
    installed_mamba_version = version("mamba-ssm")
    if installed_mamba_version != EXPECTED_MAMBA_VERSION:
        raise RuntimeError(
            f"mamba_ssm version {installed_mamba_version} is installed; "
            f"OSCD requires {EXPECTED_MAMBA_VERSION}"
        )
    # 直接使用 wheel 内的官方扩展，避免 mamba_ssm 顶层语言模型代码引入
    # 与视觉 decoder 无关的 transformers 运行时依赖。
    import selective_scan_cuda

    _SELECTIVE_SCAN_IMPORT_ERROR: Exception | None = None
except (ImportError, OSError, PackageNotFoundError, RuntimeError) as exc:  # pragma: no cover
    selective_scan_cuda = None
    _SELECTIVE_SCAN_IMPORT_ERROR = exc


class _MambaSelectiveScanFunction(torch.autograd.Function):
    """对 mamba_ssm 2.2.4 CUDA fwd/bwd 的最小 autograd 封装。"""

    @staticmethod
    def forward(ctx, u, delta, A, B, C, D, delta_bias, delta_softplus):
        tensors = (u, delta, A, B, C, D, delta_bias)
        u, delta, A, B, C, D, delta_bias = (
            tensor.contiguous() if tensor.stride(-1) != 1 else tensor
            for tensor in tensors
        )
        out, scan_intermediates, *_ = selective_scan_cuda.fwd(
            u,
            delta,
            A,
            B,
            C,
            D,
            None,
            delta_bias,
            bool(delta_softplus),
        )
        ctx.delta_softplus = bool(delta_softplus)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, scan_intermediates)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        u, delta, A, B, C, D, delta_bias, scan_intermediates = ctx.saved_tensors
        if grad_output.stride(-1) != 1:
            grad_output = grad_output.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *_ = selective_scan_cuda.bwd(
            u,
            delta,
            A,
            B,
            C,
            D,
            None,
            delta_bias,
            grad_output,
            scan_intermediates,
            None,
            None,
            ctx.delta_softplus,
            False,
        )
        return du, ddelta, dA, dB, dC, dD, ddelta_bias, None


def selective_scan_reference(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    *,
    delta_bias: torch.Tensor | None = None,
    delta_softplus: bool = True,
) -> torch.Tensor:
    """以 FP32 recurrence 实现 grouped selective scan，作为 CPU 正确性基准。

    Args:
        u/delta: ``[B, D, L]``。
        A: ``[D, N]``，其中 ``N`` 为 state dimension。
        B/C: ``[B, G, N, L]``，每个方向为一个 group。
        D/delta_bias: ``[D]``。
    """

    if u.ndim != 3 or delta.shape != u.shape:
        raise ValueError(f"u/delta must share [B,D,L], got {u.shape} and {delta.shape}")
    if A.ndim != 2 or A.shape[0] != u.shape[1]:
        raise ValueError(f"A must have shape [D,N], got {A.shape} for D={u.shape[1]}")
    if B.ndim != 4 or C.shape != B.shape:
        raise ValueError(f"B/C must share [B,G,N,L], got {B.shape} and {C.shape}")
    if B.shape[0] != u.shape[0] or B.shape[2] != A.shape[1] or B.shape[3] != u.shape[2]:
        raise ValueError("B/C dimensions are incompatible with u and A")
    if u.shape[1] % B.shape[1] != 0:
        raise ValueError("scan channels must be divisible by the number of B/C groups")

    output_dtype = u.dtype
    u32 = u.float()
    delta32 = delta.float()
    A32 = A.float()
    B32 = B.float().repeat_interleave(u.shape[1] // B.shape[1], dim=1)
    C32 = C.float().repeat_interleave(u.shape[1] // C.shape[1], dim=1)
    if delta_bias is not None:
        delta32 = delta32 + delta_bias.float().view(1, -1, 1)
    if delta_softplus:
        delta32 = F.softplus(delta32)

    state = torch.zeros(
        u.shape[0],
        u.shape[1],
        A.shape[1],
        dtype=torch.float32,
        device=u.device,
    )
    outputs: list[torch.Tensor] = []
    for index in range(u.shape[2]):
        dt = delta32[:, :, index].unsqueeze(-1)
        input_value = u32[:, :, index].unsqueeze(-1)
        state = (
            torch.exp(dt * A32.unsqueeze(0)) * state
            + dt * B32[:, :, :, index] * input_value
        )
        value = (state * C32[:, :, :, index]).sum(dim=-1)
        if D is not None:
            value = value + D.float().view(1, -1) * u32[:, :, index]
        outputs.append(value)
    return torch.stack(outputs, dim=-1).to(dtype=output_dtype)


class LayerNorm2d(nn.Module):
    """对 NCHW 特征的通道维执行 LayerNorm。"""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class FourDirectionSelectiveScan2D(nn.Module):
    """将二维特征展开为横纵正反四个序列并执行一次 selective scan。"""

    directions: Final[int] = 4

    def __init__(self, channels: int, d_state: int = 1, dt_rank: int = 1):
        super().__init__()
        if channels < 1 or d_state < 1 or dt_rank < 1:
            raise ValueError("channels, d_state and dt_rank must be positive")
        self.channels = int(channels)
        self.d_state = int(d_state)
        self.dt_rank = int(dt_rank)

        projection_dim = self.dt_rank + 2 * self.d_state
        self.x_proj_weight = nn.Parameter(
            torch.empty(self.directions, projection_dim, self.channels)
        )
        self.dt_projs_weight = nn.Parameter(
            torch.empty(self.directions, self.channels, self.dt_rank)
        )
        self.dt_projs_bias = nn.Parameter(torch.empty(self.directions, self.channels))
        self.A_logs = nn.Parameter(
            torch.log(
                torch.arange(1, self.d_state + 1, dtype=torch.float32)
                .view(1, self.d_state)
                .repeat(self.directions * self.channels, 1)
            )
        )
        self.Ds = nn.Parameter(torch.ones(self.directions * self.channels))
        self.out_norm = nn.LayerNorm(self.channels)
        self.out_proj = nn.Linear(self.channels, self.channels, bias=False)

        # 优化器据此将连续时间状态参数排除在 weight decay 之外。
        self.A_logs._no_weight_decay = True
        self.Ds._no_weight_decay = True
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.x_proj_weight)
        bound = self.dt_rank**-0.5
        nn.init.uniform_(self.dt_projs_weight, -bound, bound)
        dt = torch.exp(
            torch.rand(self.directions, self.channels)
            * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        ).clamp_min(1e-4)
        inverse_softplus = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_projs_bias.copy_(inverse_softplus)

    @staticmethod
    def cuda_backend_available() -> bool:
        return selective_scan_cuda is not None

    @staticmethod
    def cuda_backend_error() -> str | None:
        return None if _SELECTIVE_SCAN_IMPORT_ERROR is None else repr(_SELECTIVE_SCAN_IMPORT_ERROR)

    def _scan(
        self,
        xs: torch.Tensor,
        dts: torch.Tensor,
        Bs: torch.Tensor,
        Cs: torch.Tensor,
    ) -> torch.Tensor:
        batch, directions, channels, length = xs.shape
        scan_u = xs.reshape(batch, directions * channels, length)
        # CUDA kernel 要求动态输入 u/delta/B/C 同 dtype；连续时间状态参数
        # A/D/delta_bias 则始终保留 FP32，避免 AMP 降低 recurrence 稳定性。
        scan_delta = dts.to(dtype=scan_u.dtype).reshape(
            batch, directions * channels, length
        )
        Bs = Bs.to(dtype=scan_u.dtype)
        Cs = Cs.to(dtype=scan_u.dtype)
        As = -torch.exp(self.A_logs.float())
        Ds = self.Ds.float()
        delta_bias = self.dt_projs_bias.float().reshape(-1)

        if xs.is_cuda:
            if selective_scan_cuda is None:
                details = (
                    f" ({_SELECTIVE_SCAN_IMPORT_ERROR!r})"
                    if _SELECTIVE_SCAN_IMPORT_ERROR is not None
                    else ""
                )
                raise RuntimeError(
                    "OSCD CUDA execution requires the compatible mamba_ssm "
                    f"{EXPECTED_MAMBA_VERSION} selective-scan kernel; no PyTorch/CPU "
                    f"fallback is allowed{details}"
                )
            try:
                output = _MambaSelectiveScanFunction.apply(
                    scan_u,
                    scan_delta,
                    As,
                    Bs,
                    Cs,
                    Ds,
                    delta_bias,
                    True,
                )
            except Exception as exc:
                raise RuntimeError(
                    "OSCD selective-scan CUDA kernel failed. Check the pinned "
                    "mamba_ssm wheel, Torch/CUDA version and CXX11 ABI."
                ) from exc
        else:
            output = selective_scan_reference(
                scan_u,
                scan_delta,
                As,
                Bs,
                Cs,
                Ds,
                delta_bias=delta_bias,
                delta_softplus=True,
            )
        return output.reshape(batch, directions, channels, length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(
                f"FourDirectionSelectiveScan2D expects [B,{self.channels},H,W], got {x.shape}"
            )
        batch, channels, height, width = x.shape
        length = height * width
        row_sequence = x.reshape(batch, channels, length)
        column_sequence = (
            x.transpose(2, 3).contiguous().reshape(batch, channels, length)
        )
        forward_sequences = torch.stack((row_sequence, column_sequence), dim=1)
        xs = torch.cat((forward_sequences, torch.flip(forward_sequences, dims=(-1,))), dim=1)

        projected = torch.einsum("bkdl,kcd->bkcl", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(
            projected,
            [self.dt_rank, self.d_state, self.d_state],
            dim=2,
        )
        dts = torch.einsum("bkrl,kdr->bkdl", dts, self.dt_projs_weight)
        ys = self._scan(xs, dts, Bs, Cs)

        horizontal = ys[:, 0]
        horizontal_reverse = torch.flip(ys[:, 2], dims=(-1,))
        vertical = (
            ys[:, 1]
            .reshape(batch, channels, width, height)
            .transpose(2, 3)
            .contiguous()
            .reshape(batch, channels, length)
        )
        vertical_reverse = (
            torch.flip(ys[:, 3], dims=(-1,))
            .reshape(batch, channels, width, height)
            .transpose(2, 3)
            .contiguous()
            .reshape(batch, channels, length)
        )
        merged = horizontal + horizontal_reverse + vertical + vertical_reverse
        merged = merged.transpose(1, 2)
        merged = self.out_proj(self.out_norm(merged))
        return merged.transpose(1, 2).reshape(batch, channels, height, width)


class StateSpaceContextBlock(nn.Module):
    """SegMAN 风格的 CPE + SS2D + local FFN 上下文块。"""

    def __init__(self, channels: int, d_state: int = 1, ffn_ratio: int = 4):
        super().__init__()
        hidden_channels = int(channels * ffn_ratio)
        self.cpe1 = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels, bias=True
        )
        self.norm1 = LayerNorm2d(channels)
        self.scan = FourDirectionSelectiveScan2D(channels, d_state=d_state)
        self.cpe2 = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels, bias=True
        )
        self.norm2 = LayerNorm2d(channels)
        self.ffn_in = nn.Conv2d(channels, hidden_channels, 1)
        self.ffn_local = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            3,
            padding=1,
            groups=hidden_channels,
        )
        self.ffn_out = nn.Conv2d(hidden_channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.cpe1(x)
        x = x + self.scan(self.norm1(x))
        x = x + self.cpe2(x)
        hidden = F.gelu(self.ffn_in(self.norm2(x)))
        hidden = hidden + self.ffn_local(hidden)
        return x + self.ffn_out(hidden)
