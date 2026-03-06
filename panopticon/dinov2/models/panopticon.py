import torch
import torch.nn as nn
import logging
from dinov2.layers.patch_embed import make_2tuple
from torch import Tensor

logger = logging.getLogger("dinov2")
"""
任意通道输入的 Patch Embedding + 通道交叉注意力
"""


class PanopticonPE(nn.Module):
    """
    定义了一个补丁嵌入的通用类，该类接收任意通道维度的输入，并输出固定维度的嵌入。
    此类负责进行标记化和投影操作。具有属性的 self.chnfus 类则负责通道融合，即通道间的交叉注意力操作。
    """

    def __init__(
        self,
        attn_dim: int,
        embed_dim: int,
        patch_size: int,
        chnfus_cfg: dict = {},
    ):
        super().__init__()
        logger.info(f"Created PanopticonPE with attn_dim={attn_dim}, embed_dim={embed_dim}")

        self.conv3d = Conv3dWrapper(patch_size=patch_size, embed_dim=attn_dim)
        self.chnfus = ChnAttn(**chnfus_cfg, dim=attn_dim)
        self.proj = nn.Linear(attn_dim, embed_dim)

    def forward(self, x_dict: dict) -> Tensor:
        x = x_dict["imgs"]
        chn_ids = x_dict["chn_ids"]
        time_ids = x_dict.get("time_ids", None)
        feature_type_ids = x_dict.get("feature_type_ids", None)
        temporal_role_ids = x_dict.get("temporal_role_ids", None)
        polarization_ids = x_dict.get("polarization_ids", None)
        w, h = x.shape[-2:]
        mask = x_dict.get("spec_masks", None)

        x = self.conv3d(x)
        x = self.chnfus(
            x,
            chn_ids=chn_ids,
            mask=mask,
            time_ids=time_ids,
            feature_type_ids=feature_type_ids,
            temporal_role_ids=temporal_role_ids,
            polarization_ids=polarization_ids,
        )  # B,L,D
        x = self.proj(x)

        return x, h, w


class Conv3dWrapper(nn.Module):
    """Channel-wise patchification and projection, essentially wrapper around
    1 x P x P conv3d"""

    def __init__(self, patch_size, embed_dim):
        super().__init__()
        patch_size = make_2tuple(patch_size)
        patch_CHW = (1, *patch_size)
        self.conv3d = nn.Conv3d(1, embed_dim, kernel_size=patch_CHW, stride=patch_CHW)

    def forward(self, x: Tensor):
        x = self.conv3d(x.unsqueeze(1)).squeeze(1)  # B D C Hp Wp
        return x.flatten(-2).permute(0, 2, 3, 1)  # B C L D


class ChnAttn(nn.Module):
    """
    Cross attention over channels with channel embeddings to reduce any number
    of channels to a fixed dimension. Inspired by
        https://github.com/microsoft/ClimaX/blob/6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c/src/climax/arch.py#L185
    """

    def __init__(
        self,
        dim: int,
        chnemb_cfg: dict = {},
        attn_cfg: dict = {},
        layer_norm: bool = False,
        use_time_embed: bool = False,
        time_embed_num: int = 2,
        use_metadata_embed: bool = False,
        feature_type_num: int = 4,
        temporal_role_num: int = 5,
        polarization_num: int = 2,
    ):
        """
        Args:
            dim (int): Dimension of the channel attention.
            chnemb_cfg (dict): Key-value pairs for the channel embedding.
            attn_cfg (dict): Key-value pairs for the channel attention.
            layer_norm (bool, optional): Whether to apply layer norm after
                channel attention. Defaults to False.
        """
        super().__init__()

        self.chnemb = ChnEmb(**chnemb_cfg, embed_dim=dim)
        self.use_time_embed = use_time_embed
        if self.use_time_embed:
            self.time_embed = nn.Embedding(time_embed_num, dim)
        self.use_metadata_embed = use_metadata_embed
        if self.use_metadata_embed:
            # 任务特定通道语义元信息 embedding（用于层次化洪涝分割适配）。
            self.feature_type_embed = nn.Embedding(feature_type_num, dim)
            self.temporal_role_embed = nn.Embedding(temporal_role_num, dim)
            self.polarization_embed = nn.Embedding(polarization_num, dim)
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        self.xattn = CrossAttnNoQueryProj(dim=dim, **attn_cfg)

        if layer_norm:
            self.layer_norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: Tensor,
        chn_ids: Tensor,
        mask: Tensor = None,
        time_ids: Tensor = None,
        feature_type_ids: Tensor = None,
        temporal_role_ids: Tensor = None,
        polarization_ids: Tensor = None,
    ) -> Tensor:
        """
        Args:
            x (Tensor): Image tensor of shape (B, C, L, D)
            chn_ids (Tensor): Channel IDs tensor of shape (B,C) or (B,C,2) if
                stds of the SRFs curves are included, see ChnEmb.
            mask (Tensor, optional): Mask tensor of shape (B,C) indicating
                which channels have been masked out. Defaults to None.

        Returns:
            Tensor: Output tensor of shape (B, L, D) independent of the input
                channel dimension C.
        """
        B, C, L, D = x.shape

        # add embeddings
        chn_embs = self.chnemb(chn_ids)  # B,C,D
        if self.use_time_embed and time_ids is not None:
            chn_embs = chn_embs + self.time_embed(time_ids.long())
        if self.use_metadata_embed:
            # 三类元信息均提供时才加入，缺失则保持向后兼容。
            if feature_type_ids is not None and temporal_role_ids is not None and polarization_ids is not None:
                if feature_type_ids.shape != chn_ids.shape:
                    raise ValueError(
                        f"feature_type_ids shape mismatch: {tuple(feature_type_ids.shape)} vs {tuple(chn_ids.shape)}"
                    )
                if temporal_role_ids.shape != chn_ids.shape:
                    raise ValueError(
                        f"temporal_role_ids shape mismatch: {tuple(temporal_role_ids.shape)} vs {tuple(chn_ids.shape)}"
                    )
                if polarization_ids.shape != chn_ids.shape:
                    raise ValueError(
                        f"polarization_ids shape mismatch: {tuple(polarization_ids.shape)} vs {tuple(chn_ids.shape)}"
                    )
                metadata_embs = (
                    self.feature_type_embed(feature_type_ids.long())
                    + self.temporal_role_embed(temporal_role_ids.long())
                    + self.polarization_embed(polarization_ids.long())
                )
                chn_embs = chn_embs + metadata_embs
        x += chn_embs.unsqueeze(2)

        # abstract away L
        x = x.permute(0, 2, 1, 3).flatten(0, 1)  # BL,C,D
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, L, -1).flatten(0, 1)  # BL,C

        query = self.query.expand(x.shape[0], -1, -1)  # BL,1,D
        assert query.shape == (
            x.shape[0],
            1,
            x.shape[-1],
        ), f"Expected query to have shape: {x.shape[0], 1, x.shape[-1]}, but got shape: {query.shape}"

        x = self.xattn(query, x, x, key_padding_mask=mask)
        x = x.reshape(B, L, D)

        if hasattr(self, "layer_norm"):
            return self.layer_norm(x)

        return x


class ChnEmb(torch.nn.Module):

    def __init__(self, embed_dim: int, use_full_spectra=False, opt_coarsity: int = 1):
        """Creates embeddings based on the channel IDs.

        Args:
            embed_dim (int): Embedding dimension.
            use_full_spectra (bool, optional): Whether to additionally to the mean
                also use the standard deviation of optical spectral response (SRF)
                they are provided. This mode only appears in the appendix of the paper.
                Defaults to False.
            opt_coarsity (int, optional): Define the coarsity of how many nanometers
                of the mean SRF are encoded into the same embedding. Defaults to 1.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.use_full_spectra = use_full_spectra
        self.opt_coarsity = opt_coarsity

        dim1 = embed_dim // 3
        dim2 = embed_dim - 2 * dim1
        self.embed_transmit = nn.Parameter(torch.zeros(2, dim1))  # 0:V, 1:H
        self.embed_receive = nn.Parameter(torch.zeros(2, dim1))  # 0:V, 1:H
        self.embed_orbit = nn.Parameter(torch.zeros(2, dim2))  # 0:ascending, 1:descending

    def forward(self, input: Tensor) -> Tensor:

        if input.ndim == 2:  # B,C (mus)
            mus = input
        elif input.ndim == 3:  # B,C,2 (mus, sigmas)
            mus = input[:, :, 0]
        sar_indices = mus < 0
        opt_indices = torch.logical_not(sar_indices)
        device = mus.device
        dtype = self.embed_transmit.dtype

        embs = torch.zeros(list(mus.shape) + [self.embed_dim], device=device, dtype=dtype)

        # build optical embeddings

        mus[opt_indices] = (mus[opt_indices] // self.opt_coarsity).to(mus.dtype)
        if input.ndim == 2 or not self.use_full_spectra:  # only mus
            embs[opt_indices] = get_1d_sincos_pos_embed_from_grid_torch(self.embed_dim, mus[opt_indices].view(-1)).to(
                dtype
            )

        elif input.ndim == 3:  # full spectra
            mus_opt = mus[opt_indices]
            sigmas_opt = input[opt_indices][:, 1]
            embs[opt_indices] = get_1d_sincos_ipe_analytical(mus_opt, sigmas_opt, self.embed_dim, device).to(dtype)

        # build sar embeddings

        transmit = torch.cat([self.embed_transmit[0].repeat(2, 1), self.embed_transmit[1].repeat(2, 1)], dim=0).repeat(
            3, 1
        )
        receive = torch.cat(
            [
                self.embed_receive[0].unsqueeze(0),
                self.embed_receive[1].repeat(2, 1),
                self.embed_receive[0].unsqueeze(0),
            ],
            dim=0,
        ).repeat(3, 1)
        orbit = torch.stack(
            [
                self.embed_orbit.mean(axis=0),
                self.embed_orbit[0],
                self.embed_orbit[1],
            ]
        ).repeat_interleave(4, dim=0)
        sar_embs = torch.cat([transmit, receive, orbit], dim=1)

        embs[sar_indices] = sar_embs[(-(mus[sar_indices] + 1)).to(torch.int)]

        return embs


class CrossAttnNoQueryProj(nn.Module):
    """Cross Attention without query projection and final projection

    Comment: While doing the final refactor before the release, we noticed that
        we project from patches to 2304 with the conv3d and then again have the
        key & value projections from 2304 to 2304 without non-linearity in-between.
        Hence, the key & value projections are redundant and could be removed,
        significantly reducing the number of parameters. However, this is how
        the paper results were generated and we keep it for reproducibility.
        If you plan on further developing panopticon, please remove the key & value
        projections!
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.scale = head_dim**-0.5

        self.inproj_q = nn.Identity()  # no projection since query is a parameter itself
        self.inproj_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.inproj_v = nn.Linear(dim, dim, bias=qkv_bias)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, key_padding_mask=None):
        """q: (B, Nq, D), kv: (B, Nkv, D), key_padding_mask: (B, Nkv)"""

        B, Nq, D = q.shape
        q = self.inproj_q(q).reshape(B, Nq, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)
        q = q * self.scale

        B, Nkv, D = k.shape  # same as v.shape
        k = self.inproj_k(k).reshape(B, Nkv, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)
        v = self.inproj_v(v).reshape(B, Nkv, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)

        attn = q @ k.transpose(-2, -1)  # shape: (B, num_heads, Nq, Nkv)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, Nkv)
            attn = attn.masked_fill(key_padding_mask, float("-inf"))

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, D)
        return x


def get_1d_sincos_pos_embed_from_grid_torch(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb  # .double() -> unsure why the authors wanted to cast to double??


################### used in ablations


class ChnEmbSarOpt(nn.Module):
    """only learn 2 embs, one for SAR and one for optical"""

    def __init__(
        self,
        embed_dim: int,
    ):
        super().__init__()
        self.opt_embed = nn.Parameter(torch.zeros(1, embed_dim))
        self.sar_embed = nn.Parameter(torch.zeros(1, embed_dim))
        self.embed_dim = embed_dim

    def forward(self, input: Tensor):
        if input.ndim == 2:  # B,C (mus)
            mus = input
        elif input.ndim == 3:  # B,C,2 (mus, sigmas)
            mus = input[:, :, 0]

        sar_indices = mus < 0
        opt_indices = torch.logical_not(sar_indices)

        device = mus.device
        dtype = self.opt_embed.dtype
        embs = torch.zeros(list(mus.shape) + [self.embed_dim], device=device, dtype=dtype)

        embs[opt_indices] = self.opt_embed
        embs[sar_indices] = self.sar_embed
        return embs


class ChnEmbZeros(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, input):
        return torch.zeros(input.shape[0], input.shape[1], self.embed_dim, device=input.device)


################### used in appendix


def get_1d_sincos_ipe_analytical(mu: Tensor, sigma: Tensor, D: int, device, temperature=10000):

    # Create meshgrid for vectorized computation
    d_mesh = torch.arange(D, dtype=torch.float32, device=device)
    mu_mesh = mu.unsqueeze(1).expand(-1, D)
    sigma_mesh = sigma.unsqueeze(1).expand(-1, D)

    # Compute frequencies omega_i
    omega = 1.0 / (temperature ** (2 * d_mesh / D))

    # Compute the Gaussian decay term for each frequency
    # Note: We divide by sigma to normalize similar to how a Gaussian kernel would be normalized
    gaussian_term = torch.exp(-0.5 * (omega.unsqueeze(0) * sigma_mesh) ** 2)

    # Compute sine and cosine terms
    sin_term = torch.sin(omega.unsqueeze(0) * mu_mesh)
    cos_term = torch.cos(omega.unsqueeze(0) * mu_mesh)

    # Combine based on even/odd indices
    IPE = torch.where(
        d_mesh % 2 == 0, gaussian_term * sin_term, gaussian_term * cos_term  # even indices
    )  # odd indices

    return IPE
