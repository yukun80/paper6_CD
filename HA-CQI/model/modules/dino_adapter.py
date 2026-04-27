import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from .dino_meta import get_dino_arch_spec, resolve_dino_arch, resolve_extract_ids

REPO_DIR = str(Path(__file__).resolve().parents[2] / "dinov3")


class DinoV3FeatureExtractor(nn.Module):
    """冻结 DINOv3 并导出指定中间层特征。"""

    def __init__(
        self,
        dino_arch="auto",
        weights_path="dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        extract_ids=None,
        device="cuda",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.device_type = self.device.type
        if not Path(weights_path).is_file():
            raise FileNotFoundError(
                f"DINOv3 weights not found: {weights_path}. "
                "Please download the checkpoint and place it under HA-CQI/dinov3/weights/."
            )
        self.dino_arch = resolve_dino_arch(dino_arch, weights_path)
        spec = get_dino_arch_spec(self.dino_arch)
        self.model = torch.hub.load(
            REPO_DIR,
            self.dino_arch,
            source="local",
            weights=weights_path,
        )
        self.model = self.model.eval().to(self.device)
        self.embed_dim = int(spec["embed_dim"])
        self.n_layers = int(spec["num_layers"])
        self.patch_size = int(spec["patch_size"])
        self.extract_ids = resolve_extract_ids(self.dino_arch, extract_ids)

        # freeze the backbone
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x):
        scale_factor = 2 / (512 / x.shape[-1])
        x = F.interpolate(
            x, size=(512, 512), mode="bilinear", align_corners=True, antialias=True
        )
        with torch.no_grad():
            # autocast 仅在 CUDA 上开启；CPU 侧保持普通 float32 更稳妥。
            autocast_enabled = self.device_type == "cuda"
            with torch.autocast(device_type=self.device_type, dtype=torch.float16, enabled=autocast_enabled):
                feats = self.model.get_intermediate_layers(
                    x, n=range(self.n_layers), reshape=True, norm=True
                )
                feats_ = []
                for i in range(len(self.extract_ids)):
                    feats_.append(
                        F.interpolate(
                            feats[self.extract_ids[i]],
                            scale_factor=scale_factor,
                            mode="bilinear",
                        )
                    )
        return feats_


class SeparableAdapterBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, r: int = 64, act=nn.SiLU):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_dim, r, kernel_size=1, bias=False),
            nn.BatchNorm2d(r),
            act(inplace=True),
        )
        self.dw = nn.Sequential(
            nn.Conv2d(
                r, r, kernel_size=3, padding=1, groups=r, bias=False
            ),  # depthwise
            nn.BatchNorm2d(r),
            act(inplace=True),
        )
        self.proj = nn.Conv2d(r, out_dim, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.reduce(x)
        x = self.dw(x)
        x = self.proj(x)
        return x


class DinoPyramidAdapter(nn.Module):
    """将 DINOv3 中间层特征适配到与 FPN 金字塔匹配的通道数和空间尺度。

    方案A简化版：只处理 3 层 DINO 特征（跳过最浅层 raw[0]），
    分别对应 P3/P4/P5 的语义校准。每层独立 SeparableAdapterBlock + 尺度缩放。
    """

    def __init__(
        self,
        in_dim=1024,
        out_dim=256,
        bottleneck=64,
        share=False,
        num_levels=3,
    ):
        super().__init__()
        self.num_levels = num_levels
        if share:
            self.blocks = nn.ModuleList(
                [SeparableAdapterBlock(in_dim, out_dim, r=bottleneck)]
            )
        else:
            self.blocks = nn.ModuleList(
                [SeparableAdapterBlock(in_dim, out_dim, r=bottleneck) for _ in range(num_levels)]
            )
        self.share = share

    def forward(self, feats):
        """
        feats: list of num_levels tensors, each [B, C, H_i, W_i]（C = in_dim）
               默认 3 层，对应 raw[1], raw[2], raw[3]
        return: list of num_levels tensors, each [B, out_dim, S_i, S_i]
        """
        if len(feats) != self.num_levels:
            raise ValueError(
                f"DinoPyramidAdapter expects {self.num_levels} features, got {len(feats)}"
            )
        outs = []
        for i, x in enumerate(feats):
            # i=0 对应 PFF p3（scale ×1），i=1 对应 p4（×0.5），i=2 对应 p5（×0.25）
            x = F.interpolate(
                x,
                scale_factor=1.0 / (2**i),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            block = self.blocks[0] if self.share else self.blocks[i]
            outs.append(block(x))
        return outs
