# modified from https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/models/backbones/minkunet_backbone.py
from typing import List

from mmengine.registry import MODELS

from torch import Tensor, nn

from ..pv_cnn_utils import *


# equivalent to ResidualBlock in pv_cnn_utils (todo keep one of them)
class TorchSparseBasicBlock(nn.Module):
    """Torchsparse residual basic block for MinkUNet.

    Args:
        in_channels (int): In channels of block.
        out_channels (int): Out channels of block.
        kernel_size (int or Tuple[int]): Kernel_size of block.
        stride (int or Tuple[int]): Stride of the first block. Defaults to 1.
        dilation (int): Dilation of block. Defaults to 1.
        bias (bool): Whether use bias in conv. Defaults to False.
        norm_cfg (:obj:`ConfigDict` or dict): The config of normalization.
        init_cfg (:obj:`ConfigDict` or dict, optional): Initialization config.
            Defaults to None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 bias=False,
                 **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                        dilation=dilation, bias=bias),
            spnn.BatchNorm(out_channels),
            spnn.ReLU(inplace=True),
            spnn.Conv3d(
                out_channels,
                out_channels,
                kernel_size,
                stride=1,
                dilation=dilation,
                bias=bias),
            spnn.BatchNorm(out_channels)
        )

        if in_channels == out_channels and stride == 1:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(
                spnn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    dilation=dilation,
                    bias=bias),
                spnn.BatchNorm(out_channels)
            )

        self.relu = spnn.ReLU(inplace=True)

    def forward(self, x: SparseTensor) -> SparseTensor:
        out = self.relu(self.net(x) + self.downsample(x))
        return out


class TorchSparseBottleneck(nn.Module):
    """Torchsparse residual basic block for MinkUNet.

    Args:
        in_channels (int): In channels of block.
        out_channels (int): Out channels of block.
        kernel_size (int or Tuple[int]): Kernel_size of block.
        stride (int or Tuple[int]): Stride of the second block. Defaults to 1.
        dilation (int): Dilation of block. Defaults to 1.
        bias (bool): Whether use bias in conv. Defaults to False.
        norm_cfg (:obj:`ConfigDict` or dict): The config of normalization.
        init_cfg (:obj:`ConfigDict` or dict, optional): Initialization config.
            Defaults to None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 bias=False,
                 **kwargs):
        super().__init__()

        self.net = nn.Sequential(
            spnn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                dilation=dilation,
                bias=bias), spnn.BatchNorm(out_channels), spnn.ReLU(inplace=True),
            spnn.Conv3d(
                out_channels,
                out_channels,
                kernel_size,
                stride=stride,
                dilation=dilation,
                bias=bias), spnn.BatchNorm(out_channels), spnn.ReLU(inplace=True),
            spnn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                dilation=dilation,
                bias=bias), spnn.BatchNorm(out_channels))

        if in_channels == out_channels and stride == 1:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(
                spnn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    dilation=dilation,
                    bias=bias), spnn.BatchNorm(out_channels))

        self.relu = spnn.ReLU(inplace=True)

    def forward(self, x: SparseTensor) -> SparseTensor:
        out = self.relu(self.net(x) + self.downsample(x))
        return out


@MODELS.register_module()
class MinkUNetBackbone(nn.Module):
    r"""MinkUNet backbone with TorchSparse backend.

    Refer to `implementation code <https://github.com/mit-han-lab/spvnas>`_.

    Args:
        in_channels (int): Number of input voxel feature channels.
            Defaults to 4.
        base_channels (int): The input channels for first encoder layer.
            Defaults to 32.
        num_stages (int): Number of stages in encoder and decoder.
            Defaults to 4.
        encoder_channels (List[int]): Convolutional channels of each encode
            layer. Defaults to [32, 64, 128, 256].
        encoder_blocks (List[int]): Number of blocks in each encode layer.
        decoder_channels (List[int]): Convolutional channels of each decode
            layer. Defaults to [256, 128, 96, 96].
        decoder_blocks (List[int]): Number of blocks in each decode layer.
        block_type (str): Type of block in encoder and decoder.
        init_cfg (dict or :obj:`ConfigDict` or List[dict or :obj:`ConfigDict`]
            , optional): Initialization config dict.
    """
    arch_settings = {
        18: (TorchSparseBasicBlock, (2, 2, 2, 2)),
        34: (TorchSparseBasicBlock, (3, 4, 6, 3)),
        50: (TorchSparseBottleneck, (3, 4, 6, 3)),
        101: (TorchSparseBottleneck, (3, 4, 23, 3)),
        152: (TorchSparseBottleneck, (3, 8, 36, 3)),
        200: (TorchSparseBottleneck, (3, 12, 48, 3))
    }

    def __init__(self,
                 in_channels=4,
                 base_channels=32,
                 num_stages=4,
                 encoder_channels=[32, 64, 128, 256],
                 encoder_depth=34,
                 decoder_channels=[256, 128, 96, 96],
                 decoder_blocks=[2, 2, 2, 2],
                 # grid_size=None,
                 ):
        super().__init__()
        # self.grid_size = grid_size
        # assert self.grid_size is not None, 'grid_size must be provided'
        assert num_stages == len(encoder_channels) >= len(decoder_channels)
        self.num_stages = num_stages
        residual_block = self.arch_settings[encoder_depth][0]
        encoder_blocks = self.arch_settings[encoder_depth][1]

        self.conv_input = nn.Sequential(
            spnn.Conv3d(in_channels, base_channels, kernel_size=3, stride=1),
            spnn.BatchNorm(base_channels), spnn.ReLU(True),
            spnn.Conv3d(base_channels, base_channels, kernel_size=3, stride=1),
            spnn.BatchNorm(base_channels), spnn.ReLU(True))

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        encoder_channels.insert(0, base_channels)
        decoder_channels.insert(0, encoder_channels[-1])

        for i in range(num_stages):
            encoder_layer = [
                BasicConvolutionBlock(
                    encoder_channels[i],
                    encoder_channels[i],
                    ks=2,
                    stride=2,
                    dilation=1)
            ]
            for j in range(encoder_blocks[i]):
                tmp_in_channels = encoder_channels[i] if j == 0 else encoder_channels[i + 1]
                encoder_layer.append(
                    residual_block(
                        tmp_in_channels,
                        encoder_channels[i + 1],
                        ks=3,
                        stride=1,
                        dilation=1)
                )
            self.encoder.append(nn.Sequential(*encoder_layer))

            if i >= len(decoder_channels) - 1:
                continue
            decoder_layer = [
                BasicDeconvolutionBlock(
                    decoder_channels[i],
                    decoder_channels[i + 1],
                    ks=2,
                    stride=2)
            ]
            for j in range(decoder_blocks[i]):
                tmp_in_channels = decoder_channels[i + 1] + encoder_channels[-2 - i] if j == 0 \
                    else decoder_channels[i + 1]
                decoder_layer.append(
                    residual_block(
                        tmp_in_channels,
                        decoder_channels[i + 1],
                        ks=3,
                        stride=1,
                        dilation=1)
                )
            self.decoder.append(
                nn.ModuleList(
                    [decoder_layer[0],
                     nn.Sequential(*decoder_layer[1:])]))

    def forward(self, input_dict: dict) -> dict:
        """Forward function.

        Args:
            voxel_features (Tensor): Voxel features in shape (N, C).
            coors (Tensor): Coordinates in shape (N, 4),
                the columns in the order of (x_idx, y_idx, z_idx, batch_idx).

        Returns:
            Tensor: Backbone features.
        """
        x = input_dict['st_voxels']

        x = self.conv_input(x)
        laterals = [x]
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            laterals.append(x)
        laterals = laterals[:-1][::-1]

        decoder_outs = []
        for i, decoder_layer in enumerate(self.decoder):
            x = decoder_layer[0](x)

            x = torchsparse.cat((x, laterals[i]))

            x = decoder_layer[1](x)
            decoder_outs.append(x)

        input_dict['pt_feats'] = decoder_outs[-1].F
        return input_dict


@MODELS.register_module()
class MinkUNetBackboneV2(MinkUNetBackbone):
    r"""MinkUNet backbone V2.

    refer to https://github.com/PJLab-ADG/PCSeg/blob/master/pcseg/model/segmentor/voxel/minkunet/minkunet.py

    Args:
        sparseconv_backend (str): Sparse convolution backend.
    """  # noqa: E501

    def __init__(self,
                 pres,
                 vres,
                 **kwargs):
        super().__init__(**kwargs)
        self.pres = pres
        self.vres = vres

    def forward(self, input_dict: dict) -> dict:
        """Forward function.

        Args:
            voxel_features (Tensor): Voxel features in shape (N, C).
            coors (Tensor): Coordinates in shape (N, 4),
                the columns in the order of (x_idx, y_idx, z_idx, batch_idx).

        Returns:
            SparseTensor: Backbone features.
        """

        x = input_dict['st_voxels']
        # x: SparseTensor z: PointTensor
        z = PointTensor(x.F, x.C.float())

        pres = x.F.new_tensor(self.pres)
        vres = x.F.new_tensor(self.vres)
        x = initial_voxelize(z, pres, vres)

        x = self.conv_input(x)
        laterals = [x]
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            laterals.append(x)
        laterals = laterals[:-1][::-1]

        # pt_feats = []
        decoder_outs = [x]
        for i, decoder_layer in enumerate(self.decoder):
            x = decoder_layer[0](x)
            x = torchsparse.cat((x, laterals[i]))
            x = decoder_layer[1](x)
            decoder_outs.append(x)

        output = {
            "pts_backbone_fpn": decoder_outs[::-1],
            "pts_org_feats": z
        }

        pt_xyzs = []
        for decoder_out in decoder_outs[0:1]:  # only use the 1/16 pos enc
            pt_xyz = get_voxel_position(decoder_out, z.C, input_dict['points'][:, :3])
            pt_xyzs.append(pt_xyz)

        output.update({
            "pts_pos": pt_xyzs[::-1],
        })

        input_dict.update(output)
        input_dict.pop('st_voxels')
        return input_dict
