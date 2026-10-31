import torch
from torch import nn
from torch.nn import functional as F


class DSEM(nn.Module):
    def __init__(self, in_channels, out_channels=None, reduction=4,
                 kernel_sizes=(3, 5, 7)):
        super().__init__()
        out_channels = out_channels or in_channels
        inner = max(out_channels // reduction, 16)
        self.dir_convs = nn.ModuleList([
            nn.Conv2d(in_channels, inner, kernel, padding=padding, bias=False)
            for k in kernel_sizes
            for kernel, padding in [((k, 1), (k // 2, 0)), ((1, k), (0, k // 2))]
        ])
        channels = inner * len(self.dir_convs)
        groups = max(g for g in range(1, min(32, channels) + 1) if channels % g == 0)
        self.norm = nn.GroupNorm(groups, channels)
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.residual = nn.Identity() if in_channels == out_channels else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels))
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        structure = self.norm(torch.cat([conv(x) for conv in self.dir_convs], dim=1))
        return self.residual(x) + self.fusion(self.alpha.clamp(0, 1) * structure)


class DACG_Block(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4):
        super().__init__()
        if out_channels % 4:
            raise ValueError('DACG output channels must be divisible by four')
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.local_branch = nn.Conv2d(out_channels, out_channels, 1,
                                      groups=out_channels, bias=False)
        for name, k in [('short', 3), ('mid', 7), ('long', 11)]:
            setattr(self, 'strip_' + name + '_v', nn.Conv2d(
                4, out_channels, (k, 1), padding=(k // 2, 0), bias=False))
            setattr(self, 'strip_' + name + '_h', nn.Conv2d(
                1, out_channels, (1, k), padding=(0, k // 2), bias=False))
            setattr(self, 'bn_' + name, nn.BatchNorm2d(out_channels))
        self.gate_conv = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1), nn.Sigmoid())
        self.add_path = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels))
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.topo_weight = nn.Parameter(torch.tensor(0.3))

    def forward(self, x, topo_conf=None):
        projected = self.input_proj(x)
        batch, channels, height, width = projected.shape
        grouped = projected.reshape(batch, 4, channels // 4, height, width).mean(dim=2)
        branches = []
        for name in ('short', 'mid', 'long'):
            vertical = getattr(self, 'strip_' + name + '_v')(grouped)
            horizontal = getattr(self, 'strip_' + name + '_h')(vertical.mean(dim=1, keepdim=True))
            branches.append(getattr(self, 'bn_' + name)(horizontal))
        context = torch.cat(branches, dim=1)
        gate_hidden = self.gate_conv[:3](context)
        if topo_conf is not None:
            prior = F.interpolate(topo_conf, size=(height, width), mode='bilinear', align_corners=False)
            gate_hidden = gate_hidden + self.topo_weight.clamp(0, 1) * prior
        gate = self.gate_conv[3:](gate_hidden)
        fused = self.fusion(torch.cat([
            self.local_branch(projected) * gate, self.add_path(context)], dim=1))
        return self.alpha.clamp(0, 1) * fused + self.beta.clamp(0, 1) * self.shortcut(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None,
                 use_dsem=False, reduction=4):
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True))
        layers = [nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
                  nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
        if use_dsem:
            layers.append(DSEM(out_channels, reduction=reduction))
        self.conv2 = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, use_dsem=False, reduction=4):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(
            in_channels, out_channels, use_dsem=use_dsem, reduction=reduction))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, bilinear=True, use_dacg=False):
        super().__init__()
        self.bilinear = bilinear
        self.use_dacg = use_dacg
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            up_channels = in_channels
        else:
            up_channels = in_channels // 2
            self.up = nn.ConvTranspose2d(in_channels, up_channels, 2, stride=2)
        block = DACG_Block if use_dacg else DoubleConv
        self.conv = block(up_channels + skip_channels, out_channels)

    def forward(self, x, skip, topo_conf=None):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x, topo_conf) if self.use_dacg else self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv(x)
