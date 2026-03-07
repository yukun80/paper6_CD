import torch
import torch.nn as nn
import torchsparse
import torchsparse.nn as spnn
import torchsparse.nn.functional as spf
from torchsparse.nn.utils import get_kernel_offsets
from torchsparse.tensor import PointTensor, SparseTensor
from torchsparse.nn.utils import fapply


class SyncBatchNorm(nn.SyncBatchNorm):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)


class BatchNorm(nn.BatchNorm1d):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)


class BasicConvolutionBlock(nn.Module):
    def __init__(
            self,
            inc: int,
            outc: int,
            ks: int = 3,
            stride: int = 1,
            dilation: int = 1,
            if_dist: bool = False,
    ):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(
                inc, outc,
                kernel_size=ks,
                dilation=dilation,
                stride=stride,
            ),
            SyncBatchNorm(outc) if if_dist else BatchNorm(outc),
            spnn.ReLU(True),
        )

    def forward(self, x):
        out = self.net(x)
        return out


class BasicDeconvolutionBlock(nn.Module):
    def __init__(
            self,
            inc: int,
            outc: int,
            ks: int = 3,
            stride: int = 1,
            if_dist: bool = False,
    ):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(
                inc, outc,
                kernel_size=ks,
                stride=stride,
                transposed=True,
            ),
            SyncBatchNorm(outc) if if_dist else BatchNorm(outc),
            spnn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(
            self,
            inc: int,
            outc: int,
            ks: int = 3,
            stride: int = 1,
            dilation: int = 1,
            if_dist: bool = False,
    ):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(
                inc, outc,
                kernel_size=ks,
                dilation=dilation,
                stride=stride,
            ),
            SyncBatchNorm(outc) if if_dist else BatchNorm(outc),
            spnn.ReLU(True),
            spnn.Conv3d(
                outc, outc,
                kernel_size=ks,
                dilation=dilation,
                stride=1,
            ),
            SyncBatchNorm(outc) if if_dist else BatchNorm(outc),
        )
        if inc == outc * self.expansion and stride == 1:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(
                spnn.Conv3d(
                    inc, outc * self.expansion,
                    kernel_size=1,
                    dilation=1,
                    stride=stride,
                ),
                SyncBatchNorm(outc * self.expansion) if if_dist else BatchNorm(outc * self.expansion),
            )
        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
            self,
            inc: int,
            outc: int,
            ks: int = 3,
            stride: int = 1,
            dilation: int = 1,
            if_dist: bool = False,
    ):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(
                inc, outc,
                kernel_size=1,
                bias=False,
            ),
            SyncBatchNorm(outc) if if_dist else BatchNorm(outc),
            spnn.Conv3d(
                outc, outc,
                kernel_size=ks,
                stride=stride,
                bias=False,
                dilation=dilation,
            ),
            SyncBatchNorm(outc) if if_dist else BatchNorm(outc),
            spnn.Conv3d(
                outc, outc * self.expansion,
                kernel_size=1,
                bias=False,
            ),
            SyncBatchNorm(outc * self.expansion) if if_dist else BatchNorm(outc * self.expansion),
        )
        if inc == outc * self.expansion and stride == 1:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(
                spnn.Conv3d(
                    inc, outc * self.expansion,
                    kernel_size=1,
                    dilation=1,
                    stride=stride,
                ),
                SyncBatchNorm(outc * self.expansion) if if_dist else BatchNorm(outc * self.expansion),
            )
        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out


# z: PointTensor
# return: SparseTensor
def initial_voxelize(z, init_res, after_res, grid_size=None):
    new_float_coord = torch.cat([z.C[:, 0].view(-1, 1), (z.C[:, 1:4] * init_res) / after_res], 1, )
    if grid_size is not None:
        spatial_range = (int(z.C[:, 0].max().item()) + 1,) + tuple(map(int, (grid_size * init_res / after_res)))
    else:
        spatial_range = None

    pc_hash = spf.sphash(torch.floor(new_float_coord).int())
    sparse_hash = torch.unique(pc_hash)
    idx_query = spf.sphashquery(pc_hash, sparse_hash)
    counts = spf.spcount(idx_query.int(), len(sparse_hash))

    inserted_coords = spf.spvoxelize(torch.floor(new_float_coord), idx_query, counts, )
    inserted_coords = torch.round(inserted_coords).int()
    inserted_feat = spf.spvoxelize(z.F, idx_query, counts)

    new_tensor = SparseTensor(inserted_feat, inserted_coords, 1, spatial_range=spatial_range)
    # new_tensor._caches.cmaps.setdefault(new_tensor.stride, (new_tensor.coords, new_tensor.spatial_range))

    z.additional_features['idx_query'][1] = idx_query
    z.additional_features['counts'][1] = counts
    z.C = new_float_coord

    return new_tensor


# x: SparseTensor, z: PointTensor
# return: SparseTensor
def point_to_voxel(x, z):
    if (z.additional_features is None or z.additional_features.get('idx_query') is None
            or z.additional_features['idx_query'].get(x.s[0]) is None):
        pc_hash = spf.sphash(
            torch.cat([
                z.C[:, 0].int().view(-1, 1),
                torch.floor(z.C[:, 1:4] / x.s[0]).int()
            ], 1))
        sparse_hash = spf.sphash(x.C)
        idx_query = spf.sphashquery(pc_hash, sparse_hash)
        counts = spf.spcount(idx_query.int(), x.C.shape[0])
        z.additional_features['idx_query'][x.s[0]] = idx_query
        z.additional_features['counts'][x.s[0]] = counts
    else:
        idx_query = z.additional_features['idx_query'][x.s[0]]
        counts = z.additional_features['counts'][x.s[0]]

    inserted_feat = spf.spvoxelize(z.F, idx_query, counts)
    new_tensor = SparseTensor(inserted_feat, x.C, x.s)
    new_tensor._caches = x._caches

    return new_tensor


def get_voxel_position(x, C, xyz):
    pc_hash = spf.sphash(
        torch.cat([
            C[:, 0].int().view(-1, 1),
            torch.floor(C[:, 1:4] / x.s[0]).int()
        ], 1))
    sparse_hash = spf.sphash(x.C)
    idx_query = spf.sphashquery(pc_hash, sparse_hash)
    counts = spf.spcount(idx_query.int(), len(sparse_hash))

    inserted_xyz = spf.spvoxelize(xyz, idx_query, counts)
    return inserted_xyz


# x: SparseTensor, z: PointTensor
# return: PointTensor
def voxel_to_point(x, z, nearest=False):
    if (z.idx_query is None or z.weights is None or z.idx_query.get(x.s) is None
            or z.weights.get(x.s) is None):
        off = get_kernel_offsets(2, x.s, 1, device=z.F.device)
        old_hash = spf.sphash(
            torch.cat([
                z.C[:, 0].int().view(-1, 1),
                torch.floor(z.C[:, 1:4] / x.s[0]).int(),
            ], 1), off)
        pc_hash = spf.sphash(x.C.to(z.F.device))
        idx_query = spf.sphashquery(old_hash, pc_hash)
        idx_query = idx_query.transpose(0, 1).contiguous()
        weights = spf.calc_ti_weights(z.C, idx_query, scale=x.s[0])
        if nearest:
            weights[:, 1:] = 0.
            idx_query[:, 1:] = -1
        new_feat = spf.spdevoxelize(x.F, idx_query, weights)
        new_tensor = PointTensor(new_feat,
                                 z.C,
                                 idx_query=z.idx_query,
                                 weights=z.weights)
        new_tensor.additional_features = z.additional_features
        new_tensor.idx_query[x.s] = idx_query
        new_tensor.weights[x.s] = weights
        z.idx_query[x.s] = idx_query
        z.weights[x.s] = weights

    else:
        new_feat = spf.spdevoxelize(x.F, z.idx_query.get(x.s), z.weights.get(x.s))
        new_tensor = PointTensor(new_feat,
                                 z.C,
                                 idx_query=z.idx_query,
                                 weights=z.weights)
        new_tensor.additional_features = z.additional_features

    return new_tensor
