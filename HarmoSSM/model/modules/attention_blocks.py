'''
@misc{woo2018cbamconvolutionalblockattention,
      title={CBAM: Convolutional Block Attention Module}, 
      author={Sanghyun Woo and Jongchan Park and Joon-Young Lee and In So Kweon},
      year={2018},
      eprint={1807.06521},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1807.06521}, 
}
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class SAM(nn.Module):
    def __init__(self, bias=False):
        super(SAM, self).__init__()
        self.bias = bias
        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=2,
            dilation=1,
            bias=self.bias,
        )

    def forward(self, x):
        channel_max = torch.max(x, 1)[0].unsqueeze(1)
        channel_avg = torch.mean(x, 1).unsqueeze(1)
        sam_input = torch.cat((channel_max, channel_avg), dim=1).contiguous()
        # 小通道 5x5 卷积在部分 CUDA/cuDNN 组合上会触发 FIND engine 失败；
        # 仅对该空间注意力卷积绕开 cuDNN，张量和权重仍保持在 CUDA 上。
        if sam_input.is_cuda:
            with torch.backends.cudnn.flags(enabled=False):
                struct = self.conv(sam_input)
        else:
            struct = self.conv(sam_input)
        x = torch.sigmoid(struct) * x
        return x


class CAM(nn.Module):
    def __init__(self, channels, r):
        super(CAM, self).__init__()
        self.channels = channels
        self.r = r
        self.linear = nn.Sequential(
            nn.Linear(
                in_features=self.channels,
                out_features=self.channels // self.r,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features=self.channels // self.r,
                out_features=self.channels,
                bias=True,
            ),
        )

    def forward(self, x):
        channel_max = F.adaptive_max_pool2d(x, output_size=1)
        channel_avg = F.adaptive_avg_pool2d(x, output_size=1)
        b, c, _, _ = x.size()
        channel_max = self.linear(channel_max.view(b, c)).view(b, c, 1, 1)
        channel_avg = self.linear(channel_avg.view(b, c)).view(b, c, 1, 1)
        x = torch.sigmoid(channel_max + channel_avg) * x
        return x


class CBAM(nn.Module):
    def __init__(self, channels, r):
        super(CBAM, self).__init__()
        self.channels = channels
        self.r = r
        self.sam = SAM(bias=False)
        self.cam = CAM(channels=self.channels, r=self.r)

    def forward(self, x):
        output = self.cam(x)
        output = self.sam(output)
        return output + x
