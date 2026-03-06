import torch
from torch import nn
import torch.nn.functional as F

try:
    from . import resnet as models
except ImportError:  # pragma: no cover
    import model.resnet as models


class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(bin),
                    nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(reduction_dim),
                    nn.ReLU(inplace=True),
                )
            )
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode="bilinear", align_corners=True))
        return torch.cat(out, 1)


class PSPNet(nn.Module):
    def __init__(
        self,
        layers=50,
        bins=(1, 2, 3, 6),
        dropout=0.1,
        classes=3,
        zoom_factor=1,
        use_ppm=True,
        pretrained=True,
        use_aux=True,
    ):
        super(PSPNet, self).__init__()
        assert layers in [50, 101, 152]
        assert 2048 % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        self.use_aux = use_aux

        if layers == 50:
            resnet = models.resnet50(pretrained=pretrained)
        elif layers == 101:
            resnet = models.resnet101(pretrained=pretrained)
        else:
            resnet = models.resnet152(pretrained=pretrained)
        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.conv2,
            resnet.bn2,
            resnet.relu,
            resnet.conv3,
            resnet.bn3,
            resnet.relu,
            resnet.maxpool,
        )
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if "conv2" in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif "downsample.0" in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if "conv2" in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif "downsample.0" in n:
                m.stride = (1, 1)

        fea_dim = 2048
        if use_ppm:
            self.ppm = PPM(fea_dim, int(fea_dim / len(bins)), bins)
            fea_dim *= 2
        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, classes, kernel_size=1),
        )
        if self.use_aux:
            self.aux = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, classes, kernel_size=1),
            )
        else:
            self.aux = None

    def forward(self, x):
        x_size = x.size()
        h, w = x_size[2], x_size[3]

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        feat_aux = self.layer3(x)
        x = self.layer4(feat_aux)
        if self.use_ppm:
            x = self.ppm(x)
        logits = self.cls(x)
        logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=True)

        if self.training and self.aux is not None:
            aux_logits = self.aux(feat_aux)
            aux_logits = F.interpolate(aux_logits, size=(h, w), mode="bilinear", align_corners=True)
            return logits, aux_logits
        return logits


if __name__ == "__main__":
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    input = torch.rand(4, 3, 473, 473).cuda()
    model = PSPNet(
        layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=21, zoom_factor=1, use_ppm=True, pretrained=True
    ).cuda()
    model.eval()
    print(model)
    output = model(input)
    print("PSPNet", output.size())
