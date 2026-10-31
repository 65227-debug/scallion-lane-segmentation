from torch import nn

from .unet_parts import DoubleConv, Down, Up, OutConv
from .tgsm import get_tgsm


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, bilinear=True,
                 use_dsem=True, reduction=4, use_dacg=True,
                 use_tgsm=True, tgsm_version='enhanced'):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_dsem = use_dsem
        self.use_dacg = use_dacg
        self.use_tgsm = use_tgsm
        self.reduction = reduction
        self.tgsm_version = tgsm_version
        self.inc = DoubleConv(n_channels, 64, use_dsem=use_dsem, reduction=reduction)
        self.down1 = Down(64, 128, use_dsem, reduction)
        self.down2 = Down(128, 256, use_dsem, reduction)
        self.down3 = Down(256, 512, use_dsem, reduction)
        self.down4 = Down(512, 1024, use_dsem, reduction)
        if use_tgsm:
            self.tgsm = get_tgsm(tgsm_version, 1024)
        self.up1 = Up(1024, 512, 512, bilinear, use_dacg)
        self.up2 = Up(512, 256, 256, bilinear, use_dacg)
        self.up3 = Up(256, 128, 128, bilinear, use_dacg)
        self.up4 = Up(128, 64, 64, bilinear, use_dacg)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        confidence = None
        if self.use_tgsm:
            x5, confidence = self.tgsm(x5)
        x = self.up1(x5, x4, confidence)
        x = self.up2(x, x3, confidence)
        x = self.up3(x, x2, confidence)
        return self.outc(self.up4(x, x1, confidence))

    def get_config_summary(self):
        return {name: getattr(self, name) for name in (
            'n_channels', 'n_classes', 'bilinear', 'use_dsem', 'reduction',
            'use_dacg', 'use_tgsm', 'tgsm_version')}
