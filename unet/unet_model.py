"""
U-Net with DSEM + DACG + TGSM (完整版)
农田车道线分割网络

架构创新：
- 编码器: 使用 DSEM 方向感知空间增强模块
- Bottleneck: 使用 TGSM 拓扑引导空间模块
- 解码器: 使用 DACG 方向感知上下文门控模块

使用方法：
    # 基础版本
    model = UNet(n_channels=3, n_classes=3)

    # 带 DSEM
    model = UNet(n_channels=3, n_classes=3, use_dsem=True)

    # 带 DACG
    model = UNet(n_channels=3, n_classes=3, use_dacg=True)

    # 带 TGSM (需要先有DSEM或DACG)
    model = UNet(n_channels=3, n_classes=3, use_dsem=True, use_dacg=True,
                 use_tgsm=True, tgsm_version='enhanced')

    # 完整版：DSEM + DACG + TGSM
    model = UNet(n_channels=3, n_classes=3, use_dsem=True, use_dacg=True,
                 use_tgsm=True, tgsm_version='enhanced')
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# 导入现有模块
from .unet_parts import DACG_Block, DoubleConv, Down, OutConv

# 导入TGSM模块
try:
    from .tgsm import get_tgsm
except ImportError:
    # 如果在当前目录
    try:
        from tgsm import get_tgsm
    except ImportError:
        print("Warning: TGSM module not found. TGSM功能将不可用")
        get_tgsm = None


class Up_with_TGSM(nn.Module):
    """
    上采样块 - 支持TGSM的拓扑置信度引导
    """
    def __init__(self, in_channels, out_channels, bilinear=True, use_dacg=False):
        super().__init__()
        self.use_dacg = use_dacg

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            if use_dacg:
                self.conv = DACG_Block_with_TGSM(in_channels, out_channels, reduction=4)
            else:
                self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            if use_dacg:
                self.conv = DACG_Block_with_TGSM(in_channels, out_channels, reduction=4)
            else:
                self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2, topo_conf=None):
        """
        Args:
            x1: decoder特征
            x2: encoder特征
            topo_conf: 拓扑置信度图 (可选)
        """
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)

        if self.use_dacg and topo_conf is not None:
            return self.conv(x, topo_conf)
        else:
            return self.conv(x)


class DACG_Block_with_TGSM(nn.Module):
    """
    DACG块 - 支持TGSM的拓扑置信度引导
    在gate计算时加入topo_conf作为先验
    """
    def __init__(self, in_channels, out_channels, reduction=4):
        super().__init__()

        self.dacg = DACG_Block(in_channels, out_channels, reduction)

        # TGSM引导的可学习权重
        self.topo_weight = nn.Parameter(torch.tensor(0.3))

    def forward(self, x, topo_conf=None):
        if topo_conf is None:
            return self.dacg(x)

        # 使用TGSM引导
        residual = self.dacg.shortcut(x)
        x_proj = self.dacg.input_proj(x)
        x_local = self.dacg.local_branch(x_proj)

        # 多尺度方向特征
        B, C, H, W = x_proj.shape
        split_size = C // 4
        x_reshaped = x_proj.view(B, 4, split_size, H, W)
        x_strip_4ch = x_reshaped.mean(dim=2)

        feat_short_v = self.dacg.strip_short_v(x_strip_4ch)
        feat_short_1ch = feat_short_v.mean(dim=1, keepdim=True)
        feat_short = self.dacg.strip_short_h(feat_short_1ch)
        feat_short = self.dacg.bn_short(feat_short)

        feat_mid_v = self.dacg.strip_mid_v(x_strip_4ch)
        feat_mid_1ch = feat_mid_v.mean(dim=1, keepdim=True)
        feat_mid = self.dacg.strip_mid_h(feat_mid_1ch)
        feat_mid = self.dacg.bn_mid(feat_mid)

        feat_long_v = self.dacg.strip_long_v(x_strip_4ch)
        feat_long_1ch = feat_long_v.mean(dim=1, keepdim=True)
        feat_long = self.dacg.strip_long_h(feat_long_1ch)
        feat_long = self.dacg.bn_long(feat_long)

        multi_scale_feat = torch.cat([feat_short, feat_mid, feat_long], dim=1)

        # 门控调制 - 加入TGSM先验
        gate = self.dacg.gate_conv[0:3](multi_scale_feat)

        # 插值topo_conf到gate的分辨率
        topo_conf_resized = F.interpolate(
            topo_conf,
            size=gate.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        # Additive prior
        weight = torch.clamp(self.topo_weight, 0.0, 1.0)
        gate = gate + weight * topo_conf_resized

        gate = self.dacg.gate_conv[3:](gate)
        x_gated = x_local * gate

        x_add = self.dacg.add_path(multi_scale_feat)

        x_fused = torch.cat([x_gated, x_add], dim=1)
        x_out = self.dacg.fusion(x_fused)

        alpha = torch.clamp(self.dacg.alpha, 0.0, 1.0)
        beta = torch.clamp(self.dacg.beta, 0.0, 1.0)
        out = alpha * x_out + beta * residual

        return out


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False,
                 use_dsem=False, reduction=4,
                 use_dacg=False,
                 use_tgsm=False, tgsm_version='enhanced'):
        """
        U-Net with optional DSEM, DACG, and TGSM

        Args:
            n_channels: 输入图像通道数
            n_classes: 输出类别数
            bilinear: 是否使用双线性插值上采样
            use_dsem: 是否使用 DSEM (编码器)
            reduction: DSEM 通道降维比例
            use_dacg: 是否在解码器中使用 DACG 模块
            use_tgsm: 是否使用 TGSM (bottleneck)
            tgsm_version: TGSM版本 ('basic', 'enhanced', 'advanced')
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_dsem = use_dsem
        self.reduction = reduction
        self.use_dacg = use_dacg
        self.use_tgsm = use_tgsm
        self.tgsm_version = tgsm_version

        # 打印配置信息
        self._log_config()

        # ===== 编码器路径 - 使用 DSEM =====
        self.inc = DoubleConv(
            n_channels, 64,
            use_dsem=use_dsem,
            reduction=reduction
        )

        self.down1 = Down(
            64, 128,
            use_dsem=use_dsem,
            reduction=reduction
        )

        self.down2 = Down(
            128, 256,
            use_dsem=use_dsem,
            reduction=reduction
        )

        self.down3 = Down(
            256, 512,
            use_dsem=use_dsem,
            reduction=reduction
        )

        factor = 2 if bilinear else 1
        self.down4 = Down(
            512, 1024 // factor,
            use_dsem=use_dsem,
            reduction=reduction
        )

        # ===== TGSM (Bottleneck) =====
        if use_tgsm:
            if get_tgsm is None:
                raise ImportError("TGSM module not available. Please ensure tgsm.py is in the correct location.")
            self.tgsm = get_tgsm(version=tgsm_version, in_channels=1024 // factor)
            logging.info(f"  TGSM version: {tgsm_version}")

        # ===== 解码器路径（使用 DACG）=====
        self.up1 = Up_with_TGSM(1024, 512 // factor, bilinear, use_dacg=use_dacg)
        self.up2 = Up_with_TGSM(512, 256 // factor, bilinear, use_dacg=use_dacg)
        self.up3 = Up_with_TGSM(256, 128 // factor, bilinear, use_dacg=use_dacg)
        self.up4 = Up_with_TGSM(128, 64, bilinear, use_dacg=use_dacg)

        self.outc = OutConv(64, n_classes)

    def _log_config(self):
        """打印模型配置信息"""
        mode = []
        if self.use_dsem:
            mode.append("DSEM")
        if self.use_dacg:
            mode.append("DACG")
        if self.use_tgsm:
            mode.append(f"TGSM-{self.tgsm_version}")
        mode_str = " + ".join(mode) if mode else "Baseline"

        logging.info(f"\n{'='*60}")
        logging.info(f"U-Net Model Configuration")
        logging.info(f"{'='*60}")
        logging.info(f"  Mode: {mode_str}")
        logging.info(f"  Input channels: {self.n_channels}")
        logging.info(f"  Output classes: {self.n_classes}")
        logging.info(f"  Bilinear upsampling: {self.bilinear}")
        if self.use_dsem:
            logging.info(f"  DSEM reduction: {self.reduction}")
        if self.use_dacg:
            logging.info(f"  Using DACG in decoder")
        if self.use_tgsm:
            logging.info(f"  Using TGSM ({self.tgsm_version}) at bottleneck")
        logging.info(f"{'='*60}\n")

    def forward(self, x):
        # ===== Encoder =====
        x1 = self.inc(x)      # [B, 64, H, W]
        x2 = self.down1(x1)   # [B, 128, H/2, W/2]
        x3 = self.down2(x2)   # [B, 256, H/4, W/4]
        x4 = self.down3(x3)   # [B, 512, H/8, W/8]
        x5 = self.down4(x4)   # [B, 1024, H/16, W/16]

        # ===== TGSM (Bottleneck) =====
        if self.use_tgsm:
            topo_feat, topo_conf = self.tgsm(x5)
            x5 = topo_feat  # 使用TGSM增强的特征
        else:
            topo_conf = None

        # ===== Decoder (with TGSM guidance) =====
        x = self.up1(x5, x4, topo_conf)
        x = self.up2(x, x3, topo_conf)
        x = self.up3(x, x2, topo_conf)
        x = self.up4(x, x1, topo_conf)

        logits = self.outc(x)
        return logits

    def get_config_summary(self):
        """获取模型配置摘要"""
        config = {
            'n_channels': self.n_channels,
            'n_classes': self.n_classes,
            'bilinear': self.bilinear,
            'use_dsem': self.use_dsem,
            'reduction': self.reduction,
            'use_dacg': self.use_dacg,
            'use_tgsm': self.use_tgsm,
            'tgsm_version': self.tgsm_version if self.use_tgsm else None
        }
        return config

    def use_checkpointing(self):
        """启用梯度检查点以节省内存"""
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        if self.use_tgsm:
            self.tgsm = torch.utils.checkpoint(self.tgsm)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


# ===== 测试代码 =====
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    print("=" * 70)
    print("U-Net with DSEM + DACG + TGSM Test")
    print("=" * 70)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    x = torch.randn(2, 3, 256, 256)

    # 测试1: Baseline
    print("\n[Test 1] Baseline U-Net")
    model_baseline = UNet(n_channels=3, n_classes=3)
    out = model_baseline(x)
    print(f"✓ Output: {out.shape}, Params: {count_parameters(model_baseline):.2f}M")

    # 测试2: DSEM + DACG
    print("\n[Test 2] U-Net + DSEM + DACG")
    model_current = UNet(n_channels=3, n_classes=3, use_dsem=True, use_dacg=True)
    out = model_current(x)
    print(f"✓ Output: {out.shape}, Params: {count_parameters(model_current):.2f}M")

    # 测试3: DSEM + DACG + TGSM
    print("\n[Test 3] U-Net + DSEM + DACG + TGSM (Enhanced)")
    model_full = UNet(n_channels=3, n_classes=3,
                     use_dsem=True, use_dacg=True,
                     use_tgsm=True, tgsm_version='enhanced')
    out = model_full(x)
    print(f"✓ Output: {out.shape}, Params: {count_parameters(model_full):.2f}M")
    print(f"  Increase: +{(count_parameters(model_full) - count_parameters(model_current)):.2f}M")

    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)
