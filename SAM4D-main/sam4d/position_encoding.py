import math
import torch
from torch import nn
from typing import Any, Optional, Tuple, Type, List


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention Is All You Need paper, generalized to work on images.
    """

    def __init__(
            self,
            num_pos_feats,
            temperature: int = 10000,
            normalize: bool = True,
            scale: Optional[float] = None,
    ):
        super().__init__()
        assert num_pos_feats % 2 == 0, "Expecting even model width"
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

        self.cache = {}

    def encode_xy(self, xy):
        # The positions are expected to be normalized
        assert xy.shape[-1] == 2
        assert xy.min() >= 0 and xy.max() <= 1
        x_embed = xy[..., 0] * self.scale
        y_embed = xy[..., 1] * self.scale

        dim_t = torch.arange(self.num_pos_feats // 2, dtype=torch.float32, device=xy.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats // 2)

        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        return torch.cat((pos_y, pos_x), dim=-1)  # (..., 256)

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        assert x.ndim == 4, f'Expecting 4D tensor, got {x.ndim}D tensor'
        cache_key = (x.shape[-2], x.shape[-1])
        if cache_key in self.cache:
            return self.cache[cache_key][None].repeat(x.shape[0], 1, 1, 1)
        y_embed = (
            (torch.arange(0, x.shape[-2], dtype=torch.float32, device=x.device) + 0.5)  # shift to pixel center
            .view(1, -1, 1)
            .repeat(x.shape[0], 1, x.shape[-1])
        )
        x_embed = (
            (torch.arange(0, x.shape[-1], dtype=torch.float32, device=x.device) + 0.5)  # shift to pixel center
            .view(1, 1, -1)
            .repeat(x.shape[0], x.shape[-2], 1)
        )

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (x.shape[-2] + eps) * self.scale
            x_embed = x_embed / (x.shape[-1] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats // 2, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats // 2)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        self.cache[cache_key] = pos[0]
        return pos


def init_t_xy(end_x: int, end_y: int):
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode="floor").float()
    return t_x, t_y


def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 10000.0):
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    t_x, t_y = init_t_xy(end_x, end_y)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[-2], x.shape[-1])
    shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_enc(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
        repeat_freqs_k: bool = False,
):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = (
        torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        if xk.shape[-2] != 0
        else None
    )
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    if xk_ is None:
        # no keys to rotate, due to dropout
        return xq_out.type_as(xq).to(xq.device), xk
    # repeat freqs along seq_len dim to match k seq_len
    if repeat_freqs_k:
        r = xk_.shape[-2] // xq_.shape[-2]
        if freqs_cis.is_cuda:
            freqs_cis = freqs_cis.repeat(*([1] * (freqs_cis.ndim - 2)), r, 1)
        else:
            # torch.repeat on complex numbers may not be supported on non-CUDA devices
            # (freqs_cis has 4 dims and we repeat on dim 2) so we use expand + flatten
            freqs_cis = freqs_cis.unsqueeze(2).expand(-1, -1, r, -1, -1).flatten(2, 3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


class PositionEmbeddingRandomAbsolute(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    todo this may not right for pe using absolute coords, it can not make sure 1 position has unique embedding
    """

    def __init__(self, num_pos_feats: int = 64, num_in_channels: int = 3, scale: Optional[float] = None, pc_range=None) -> None:
        super().__init__()
        self.num_pos_feats = num_pos_feats
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((num_in_channels, num_pos_feats // 2)),
        )
        self.pc_range = pc_range

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = coords @ self.positional_encoding_gaussian_matrix
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward_with_coords(self, coords_input: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        return self._pe_encoding(coords.to(torch.float))  # B x N x C


class PositionEmbeddingMLP(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention Is All You Need paper, generalized to work on images.
    """

    def __init__(
            self,
            num_pos_feats,
            lss_dbound,
            image_size,
            backbone_stride,
            add_layernorm=False,
    ):
        super().__init__()

        self.num_pos_feats = num_pos_feats
        self.lss_dbound = lss_dbound
        self.image_size = [image_size, image_size] if isinstance(image_size, int) else image_size
        self.backbone_stride = backbone_stride

        self.frustum = self.create_frustum()  # H W D 4
        self.depth_num = self.frustum.shape[2]

        if not add_layernorm:
            self.mlp_embedding = nn.Sequential(
                nn.Linear(self.depth_num * 3, self.num_pos_feats * 4),
                nn.ReLU(inplace=True),
                nn.Linear(self.num_pos_feats * 4, self.num_pos_feats)
            )
        else:
            self.mlp_embedding = nn.Sequential(
                nn.Linear(self.depth_num * 3, self.num_pos_feats * 4),
                nn.ReLU(inplace=True),
                nn.Linear(self.num_pos_feats * 4, self.num_pos_feats),
                nn.LayerNorm(self.num_pos_feats)
            )

    def create_frustum(self):
        iH, iW = self.image_size
        fH, fW = iH // self.backbone_stride, iW // self.backbone_stride

        ds = torch.arange(*self.lss_dbound, dtype=torch.float).view(1, 1, -1).expand(fH, fW, -1)
        _, _, D = ds.shape

        xs = torch.arange(0, fW, dtype=torch.float) + 0.5  # Shift to center of pixel, keep same with PositionEmbeddingSine
        xs = (xs * self.backbone_stride).view(1, fW, 1).expand(fH, fW, D)
        ys = torch.arange(0, fH, dtype=torch.float) + 0.5  # Shift to center of pixel, keep same with PositionEmbeddingSine
        ys = (ys * self.backbone_stride).view(fH, 1, 1).expand(fH, fW, D)

        ones = ds.new_ones(*ds.shape)  # make frustum homogeneous
        frustum = torch.stack((xs, ys, ds, ones), -1)

        return nn.Parameter(frustum, requires_grad=False)

    def embed_point(self, points: torch.Tensor):
        origin_shape = points.shape[:-1]
        assert points.shape[-1] == 3, f'points.shape: {points.shape}'
        points = points.unsqueeze(-2).expand(*origin_shape, self.depth_num, 3).reshape(-1, self.depth_num * 3)

        pos_enc = self.mlp_embedding(points).reshape(*origin_shape, -1)
        return pos_enc

    def embed_image(self, meta: dict, uv=None, T_rel=None):
        if uv is None:
            points = self.frustum  # (H, W, D, 4)
        else:
            assert uv.ndim == 3, f'uv.shape: {uv.shape}'
            assert uv.shape[-1] == 2, f'uv.shape: {uv.shape}'
            h, w = uv.shape[:2]
            ds = torch.arange(*self.lss_dbound, dtype=torch.float).view(1, 1, -1, 1).expand(h, w, -1, 1).to(uv.device)
            ones = ds.new_ones(*ds.shape)  # make frustum homogeneous
            uv = uv.view(h, w, 1, 2).expand(h, w, ds.shape[2], 2)
            points = torch.concat((uv, ds, ones), -1)

        img_aug_matrix = points.new_tensor(meta['img_aug_matrix'])
        camera2lidar = points.new_tensor(meta['camera2lidar'])
        camera_intrinsic = points.new_tensor(meta['camera_intrinsics'])
        lidar_aug_matrix = points.new_tensor(meta['lidar_aug_matrix'])

        # inverse image aug
        img_aug_matrix_inv = torch.inverse(img_aug_matrix).view(1, 1, 1, 4, 4)
        points = img_aug_matrix_inv.matmul(points.unsqueeze(-1)).squeeze(-1)  # (H, W, D, 4)

        points = torch.cat([points[..., :2] * points[..., 2:3], points[..., 2:]], -1)
        cam_intrinsic_inv = torch.inverse(camera_intrinsic).view(1, 1, 1, 4, 4)
        points = cam_intrinsic_inv.matmul(points.unsqueeze(-1)).squeeze(-1)  # (H, W, D, 4)

        # camera to lidar
        camera2lidar = lidar_aug_matrix.matmul(camera2lidar).view(1, 1, 1, 4, 4)
        points = camera2lidar.matmul(points.unsqueeze(-1))  # (H, W, D, 4, 1)
        if T_rel is not None:
            T_rel = T_rel.to(points.device).to(points.dtype).view(1, 1, 1, 4, 4)
            points = T_rel.matmul(points)  # (H, W, D, 4, 1)

        points = points.squeeze(-1)[..., :3]  # (H, W, D, 3)

        H, W, D, _ = points.shape
        points = points.reshape(H * W, -1)  # (HW, D*3)

        pos_enc = self.mlp_embedding(points).reshape(H, W, -1)
        if uv is not None:
            return pos_enc
        else:
            return pos_enc.permute(2, 0, 1)


class UnionPositionEmbedding(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention Is All You Need paper, generalized to work on images.
    """

    def __init__(
            self,
            modal: dict,
            # img_pe_cfg: dict = None,
            # pts_pe_cfg: dict = None,
    ):
        super().__init__()

        self.img_pe_layer = None
        if 'img' in modal:
            self.img_pe_layer = PositionEmbeddingSine(**modal['img']['pe_cfg'])

        self.pts_pe_layer = None
        if 'pts' in modal:
            pts_pe_cfg = modal['pts']['pe_cfg']
            self.pts_pe_layer = PositionEmbeddingRandomAbsolute(**pts_pe_cfg)

        '''
        support MLP spatial embedding.
        for image, each pixel to get n frustum points, and nx3 coords to pass a MLP to get 256 dim feat.
        for voxel, each point to repeat n times, and nx3 coords to pass a MLP to get 256 dim feat.
        for point prompt, first project it to image, do image embedding & union embedding, then do pts embedding and union embedding, sum. 
        '''
        self.union_pe_layer = None
        if 'img' in modal and 'pts' in modal:
            self.union_pe_layer = PositionEmbeddingMLP(**modal['pts']['union_pe_cfg'])

    def get_image_dense_pe(self, x: torch.Tensor):
        # x NCHW
        return self.img_pe_layer.forward(x)

    def get_image_sparse_pe(self, x: torch.Tensor):
        # x: (..., 2) 2 is uv normalized coords
        return self.img_pe_layer.encode_xy(x)

    def get_point_pe(self, x: torch.Tensor):
        # x: (..., 3) 3 is xyz absolute coords
        return self.pts_pe_layer.forward_with_coords(x)

    def get_union_image_pe(self, meta: dict, uv=None, T_rel=None):
        # uv: (..., 2) 2 is absolute image coords
        return self.union_pe_layer.embed_image(meta, uv, T_rel)

    def get_union_point_pe(self, pts: torch.Tensor):
        return self.union_pe_layer.embed_point(pts)
