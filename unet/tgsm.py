import torch
from torch import nn
from torch.nn import functional as F


class EnhancedTopologyConfidence(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.register_buffer('skeleton_kernel', torch.tensor(
            [[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=torch.float32).reshape(1, 1, 3, 3) / 18)
        self.register_buffer('connect_kernel', torch.tensor(
            [[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=torch.float32).reshape(1, 1, 3, 3) / 8)

    def forward(self, x):
        mean = x.mean(dim=(2, 3), keepdim=True)
        square = (x - mean).square()
        energy = square / (square.mean(dim=(2, 3), keepdim=True) + self.eps)
        stable = 1 / (1 + energy.mean(dim=1, keepdim=True))
        channel_mean = x.mean(dim=1, keepdim=True)
        skeleton = torch.sigmoid(2 * F.conv2d(channel_mean, self.skeleton_kernel, padding=1))
        connected = torch.sigmoid(F.conv2d(channel_mean, self.connect_kernel, padding=1) - 0.3)
        confidence = stable * skeleton * connected
        return F.avg_pool2d(F.pad(confidence, (1, 2, 0, 0), mode='replicate'), (1, 4), stride=1)


class MultiScaleDirectionalAggregation(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.scales = (1, 2, 4)
        widths = [channels // 3 + int(i < channels % 3) for i in range(3)]
        self.proj_h = nn.ModuleList([nn.Conv2d(channels, c, 1, bias=False) for c in widths])
        self.proj_w = nn.ModuleList([nn.Conv2d(channels, c, 1, bias=False) for c in widths])
        self.fusion = nn.Sequential(nn.Conv2d(channels * 2, channels, 1, bias=False),
                                    nn.BatchNorm2d(channels), nn.ReLU(inplace=True))

    def forward(self, x):
        height, width = x.shape[-2:]
        horizontal, vertical = [], []
        for i, scale in enumerate(self.scales):
            kh, kw = min(scale, height), min(scale, width)
            h = F.avg_pool2d(x, (1, kw), stride=(1, kw)).mean(dim=3, keepdim=True)
            w = F.avg_pool2d(x, (kh, 1), stride=(kh, 1)).mean(dim=2, keepdim=True)
            horizontal.append(F.interpolate(self.proj_h[i](h), (height, width), mode='bilinear', align_corners=False))
            vertical.append(F.interpolate(self.proj_w[i](w), (height, width), mode='bilinear', align_corners=False))
        return self.fusion(torch.cat(horizontal + vertical, dim=1))


class EnhancedTGSM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.dir_agg = MultiScaleDirectionalAggregation(in_channels)
        self.topo_conf = EnhancedTopologyConfidence()
        self.proj = nn.Conv2d(1, in_channels, 1)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        confidence = self.topo_conf(self.dir_agg(x))
        return x + self.alpha.clamp(0, 1) * self.proj(confidence), confidence


def get_tgsm(version='enhanced', in_channels=1024):
    if version != 'enhanced':
        raise ValueError('Supported TGSM version: enhanced')
    return EnhancedTGSM(in_channels)
