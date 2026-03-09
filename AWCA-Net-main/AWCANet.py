import os
from pathlib import Path
from typing import Dict, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

if __package__:
    # 兼容包内导入（from xxx import AWCANet）
    from .backbones.pvtv2 import pvt_v2_b2
    from .NECM import NECM
    from .MSAWM import MSAWM
else:
    # 兼容直接脚本运行（python AWCA-Net-main/train_awca.py）
    from backbones.pvtv2 import pvt_v2_b2
    from NECM import NECM
    from MSAWM import MSAWM


def _strip_prefix_if_needed(state_dict: Dict[str, torch.Tensor], prefixes: Iterable[str]) -> Dict[str, torch.Tensor]:
    """清理常见权重前缀，提升不同来源 checkpoint 的兼容性。"""
    for prefix in prefixes:
        if all(k.startswith(prefix) for k in state_dict.keys()):
            return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Encoder(nn.Module):
    def __init__(self, pvt_path: str | os.PathLike | None = None):
        super(Encoder, self).__init__()
        self.pvt = pvt_v2_b2()
        if pvt_path is not None:
            self.load_pretrained(pvt_path)

    def load_pretrained(self, pvt_path: str | os.PathLike) -> None:
        """
        加载 PVTv2-B2 预训练权重。
        仅覆盖 backbone 中同名参数，避免因来源差异导致严格加载失败。
        """
        pvt_path = Path(pvt_path)
        if not pvt_path.exists():
            raise FileNotFoundError(f"PVT checkpoint not found: {pvt_path}")

        ckpt = torch.load(str(pvt_path), map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        if not isinstance(ckpt, dict):
            raise ValueError(f"Unexpected checkpoint format at {pvt_path}")

        ckpt = _strip_prefix_if_needed(ckpt, prefixes=("module.", "backbone."))
        model_dict = self.pvt.state_dict()
        matched = {k: v for k, v in ckpt.items() if k in model_dict and v.shape == model_dict[k].shape}
        if not matched:
            raise RuntimeError(f"No matching keys found when loading {pvt_path}")
        model_dict.update(matched)
        self.pvt.load_state_dict(model_dict)
        print(f"[AWCANet] Loaded {len(matched)} backbone params from {pvt_path}")

    def forward(self, a, b):
        pvt_a = self.pvt(a)
        pvt_b = self.pvt(b)
        return pvt_a, pvt_b


class Decoder(nn.Module):
    def __init__(self, channels, num_classes):
        super(Decoder, self).__init__()
        self.enhance = NECM()
        self.conv1 = BasicConv2d(2 * 64, 64, 1)
        self.conv2 = BasicConv2d(2 * 128, 128, 1)
        self.conv3 = BasicConv2d(2 * 320, 320, 1)
        self.conv4 = BasicConv2d(2 * 512, 512, 1)

        self.conv_4 = BasicConv2d(channels[0], channels[0], 3, 1, 1)
        self.conv_3 = BasicConv2d(channels[1], channels[1], 3, 1, 1)
        self.conv_2 = BasicConv2d(channels[2], channels[2], 3, 1, 1)
        self.conv_1 = BasicConv2d(channels[3], channels[3], 3, 1, 1)

        self.frh4 = nn.Conv2d(channels[0], num_classes, 1)
        self.frh3 = nn.Conv2d(channels[1], num_classes, 1)
        self.frh2 = nn.Conv2d(channels[2], num_classes, 1)
        self.frh1 = nn.Conv2d(channels[3], num_classes, 1)

        self.lm = MSAWM(
            channels=channels,
            kernel_sizes=[1, 3, 5],
            expansion_factor=6,
            dw_parallel=True,
            add=True,
            lgag_ks=3,
            activation="relu6",
        )

    def forward(self, pvt_a, pvt_b):
        pvt_a1, pvt_a2, pvt_a3, pvt_a4 = self.enhance(pvt_a)
        pvt_b1, pvt_b2, pvt_b3, pvt_b4 = self.enhance(pvt_b)

        layer_1 = self.conv_1(self.conv1(torch.cat((pvt_a1, pvt_b1), dim=1)))
        layer_2 = self.conv_2(self.conv2(torch.cat((pvt_a2, pvt_b2), dim=1)))
        layer_3 = self.conv_3(self.conv3(torch.cat((pvt_a3, pvt_b3), dim=1)))
        layer_4 = self.conv_4(self.conv4(torch.cat((pvt_a4, pvt_b4), dim=1)))
        outs = self.lm(layer_4, [layer_3, layer_2, layer_1])

        u4 = self.frh4(outs[0])
        u3 = self.frh3(outs[1])
        u2 = self.frh2(outs[2])
        u1 = self.frh1(outs[3])

        u4 = F.interpolate(u4, scale_factor=32, mode="bilinear")
        u3 = F.interpolate(u3, scale_factor=16, mode="bilinear")
        u2 = F.interpolate(u2, scale_factor=8, mode="bilinear")
        u1 = F.interpolate(u1, scale_factor=4, mode="bilinear")

        # AMP 训练下采用 logits + BCEWithLogits 更稳定，故此处返回 raw logits。
        return [u4, u3, u2, u1]


class AWCANet(nn.Module):
    def __init__(self, pvt_path: str | os.PathLike | None = None):
        super(AWCANet, self).__init__()
        self.encoder = Encoder(pvt_path=pvt_path)
        self.decoder = Decoder(channels=[512, 320, 128, 64], num_classes=1)

    def forward(self, a, b):
        pvt_a, pvt_b = self.encoder(a, b)
        pred = self.decoder(pvt_a, pvt_b)
        return pred


if __name__ == "__main__":
    a = torch.rand(2, 3, 256, 256).cuda()
    b = torch.rand(2, 3, 256, 256).cuda()
    model = AWCANet(pvt_path=None).cuda()
    outs = model(a, b)
    print([o.shape for o in outs])
