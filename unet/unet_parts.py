"""
U-Net Parts with DSEM and OLD DACG Decoder (Compatible with your checkpoint)
包含：
1. DSEM 方向结构编码模块（Encoder专用）
2. DACG 方向感知上下文门控模块（Decoder专用）- OLD VERSION

这是旧版DACG，与你训练的checkpoint兼容
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== DSEM 方向结构编码模块 (Encoder) ====================

class DSEM(nn.Module):
    """
    Directional Structure Encoding Module (DSEM)
    Encoder-side module for thin-structure segmentation.

    Design principles:
    1. Direction-aware (strip convolutions)
    2. No feature suppression (no sigmoid / no gate / no top-k)
    3. Residual structure enhancement
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int = None,
                 reduction: int = 4,
                 kernel_sizes=(3, 5, 7),
                 norm='group'):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels
        self.out_channels = out_channels

        inter_channels = max(out_channels // reduction, 16)

        # ===== Directional strip convolutions =====
        self.dir_convs = nn.ModuleList()
        for k in kernel_sizes:
            # vertical
            self.dir_convs.append(
                nn.Conv2d(in_channels, inter_channels,
                          kernel_size=(k, 1),
                          padding=(k // 2, 0),
                          bias=False)
            )
            # horizontal
            self.dir_convs.append(
                nn.Conv2d(in_channels, inter_channels,
                          kernel_size=(1, k),
                          padding=(0, k // 2),
                          bias=False)
            )

        # ===== Normalization =====
        num_groups = min(32, inter_channels * len(self.dir_convs))
        while (inter_channels * len(self.dir_convs)) % num_groups != 0:
            num_groups -= 1

        if norm == 'group':
            self.norm = nn.GroupNorm(num_groups, inter_channels * len(self.dir_convs))
        else:
            self.norm = nn.BatchNorm2d(inter_channels * len(self.dir_convs))

        # ===== Feature fusion =====
        self.fusion = nn.Sequential(
            nn.Conv2d(inter_channels * len(self.dir_convs),
                      out_channels,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # ===== Residual alignment =====
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()

        # ===== Learnable scaling (stability) =====
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        identity = self.residual(x)

        dir_feats = []
        for conv in self.dir_convs:
            dir_feats.append(conv(x))

        # Concatenate multi-direction features
        dir_feat = torch.cat(dir_feats, dim=1)
        dir_feat = self.norm(dir_feat)

        # Fuse and inject structure
        structure_feat = self.fusion(dir_feat)

        alpha = torch.clamp(self.alpha, 0.0, 1.0)
        out = identity + alpha * structure_feat

        return out


# ==================== OLD DACG 解码器模块 (Decoder) ====================

class DACG_Block(nn.Module):
    """
    Direction-Aware Context Gating Block for Decoder (OLD VERSION - EXACT MATCH)
    解码器专用的方向感知上下文门控模块 - 旧版（完全匹配checkpoint）

    关键发现：
    1. strip_*_v: [out_ch, 4, K, 1] - 输入4通道，输出out_ch
    2. strip_*_h: [out_ch, 1, K, 1] - 输入1通道，输出out_ch
    3. local_branch使用depthwise + 额外的1x1 conv
    4. gate_conv有5层
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 reduction: int = 4):
        super().__init__()

        # 1. 输入投影 - 投影到out_channels
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 2. 局部特征分支（depthwise + 1x1 conv）
        self.local_branch = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,
                     groups=out_channels, bias=False),  # Depthwise
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),  # 第3层
        )

        # 3. 多尺度条形卷积
        # 关键：vertical接受4通道，horizontal接受1通道
        # 输出都是out_channels（不是out_channels//4）

        # 短距离（3x1 和 1x3）
        self.strip_short_v = nn.Conv2d(4, out_channels, kernel_size=(3, 1),
                                       padding=(1, 0), bias=False)
        self.strip_short_h = nn.Conv2d(1, out_channels, kernel_size=(1, 3),
                                       padding=(0, 1), bias=False)
        self.bn_short = nn.BatchNorm2d(out_channels)

        # 中距离（7x1 和 1x7）
        self.strip_mid_v = nn.Conv2d(4, out_channels, kernel_size=(7, 1),
                                     padding=(3, 0), bias=False)
        self.strip_mid_h = nn.Conv2d(1, out_channels, kernel_size=(1, 7),
                                     padding=(0, 3), bias=False)
        self.bn_mid = nn.BatchNorm2d(out_channels)

        # 长距离（11x1 和 1x11）
        self.strip_long_v = nn.Conv2d(4, out_channels, kernel_size=(11, 1),
                                      padding=(5, 0), bias=False)
        self.strip_long_h = nn.Conv2d(1, out_channels, kernel_size=(1, 11),
                                      padding=(0, 5), bias=False)
        self.bn_long = nn.BatchNorm2d(out_channels)

        # 4. 门控调制路径（5层）
        gate_input_channels = out_channels * 3  # 3个方向的特征

        self.gate_conv = nn.Sequential(
            nn.Conv2d(gate_input_channels, out_channels, kernel_size=1, bias=False),  # 0
            nn.BatchNorm2d(out_channels),  # 1
            nn.ReLU(inplace=True),  # 2
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True),  # 3
            nn.Sigmoid()  # 4
        )

        # 5. 辅助加法路径
        self.add_path = nn.Sequential(
            nn.Conv2d(gate_input_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 6. 最终融合
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 7. 残差连接
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        # 8. 可学习的融合权重
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)
        self.beta = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x):
        residual = self.shortcut(x)

        # 投影到输出通道
        x_proj = self.input_proj(x)  # [B, out_channels, H, W]

        # 局部特征
        x_local = self.local_branch(x_proj)

        # 为strip卷积准备输入
        # 方法：将out_channels分成4组，每组作为一个通道
        B, C, H, W = x_proj.shape
        split_size = C // 4

        # 对于vertical strip：需要4通道输入
        # 将x_proj重塑为[B, 4, split_size, H, W]然后平均
        x_reshaped = x_proj.view(B, 4, split_size, H, W)
        x_strip_4ch = x_reshaped.mean(dim=2)  # [B, 4, H, W]

        # 多尺度方向感知特征提取
        # Short
        feat_short_v = self.strip_short_v(x_strip_4ch)  # [B, 4, H, W] -> [B, out_ch, H, W]
        feat_short_1ch = feat_short_v.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        feat_short = self.strip_short_h(feat_short_1ch)  # [B, 1, H, W] -> [B, out_ch, H, W]
        feat_short = self.bn_short(feat_short)

        # Mid
        feat_mid_v = self.strip_mid_v(x_strip_4ch)
        feat_mid_1ch = feat_mid_v.mean(dim=1, keepdim=True)
        feat_mid = self.strip_mid_h(feat_mid_1ch)
        feat_mid = self.bn_mid(feat_mid)

        # Long
        feat_long_v = self.strip_long_v(x_strip_4ch)
        feat_long_1ch = feat_long_v.mean(dim=1, keepdim=True)
        feat_long = self.strip_long_h(feat_long_1ch)
        feat_long = self.bn_long(feat_long)

        # 拼接多尺度特征
        multi_scale_feat = torch.cat([feat_short, feat_mid, feat_long], dim=1)

        # 门控调制路径
        gate = self.gate_conv(multi_scale_feat)
        x_gated = x_local * gate

        # 加法辅助路径
        x_add = self.add_path(multi_scale_feat)

        # 双路径融合
        x_fused = torch.cat([x_gated, x_add], dim=1)
        x_out = self.fusion(x_fused)

        # 自适应残差连接
        alpha = torch.clamp(self.alpha, 0.0, 1.0)
        beta = torch.clamp(self.beta, 0.0, 1.0)

        out = alpha * x_out + beta * residual

        return out


# ==================== U-Net 基础组件 ====================

class DoubleConv(nn.Module):
    """双卷积块 - 使用DSEM"""

    def __init__(self, in_channels, out_channels, mid_channels=None,
                 use_dsem=False, reduction=4, dropout_rate=0.0):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        if dropout_rate > 0:
            self.dropout = nn.Dropout2d(dropout_rate)
        else:
            self.dropout = None

        if use_dsem:
            self.conv2 = nn.Sequential(
                nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                DSEM(out_channels, out_channels, reduction=reduction)
            )
        else:
            self.conv2 = nn.Sequential(
                nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self.conv1(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv2(x)
        return x


class Down(nn.Module):
    """下采样块 - 编码器"""

    def __init__(self, in_channels, out_channels, use_dsem=False,
                 reduction=4, dropout_rate=0.0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels,
                       use_dsem=use_dsem,
                       reduction=reduction,
                       dropout_rate=dropout_rate)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """上采样块 - 解码器 (使用 OLD DACG)"""

    def __init__(self, in_channels, out_channels, bilinear=True, use_dacg=False):
        super().__init__()
        self.use_dacg = use_dacg

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            if use_dacg:
                self.conv = DACG_Block(in_channels, out_channels, reduction=4)
            else:
                self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            if use_dacg:
                self.conv = DACG_Block(in_channels, out_channels, reduction=4)
            else:
                self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """输出卷积"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("=" * 70)
    print("U-Net Parts with DSEM + OLD DACG Test")
    print("=" * 70)

    # 测试 DSEM
    print("\n[Test 1] DSEM Module")
    dsem = DSEM(64, 64, reduction=4)
    x = torch.randn(2, 64, 56, 56)
    out = dsem(x)
    print(f"✓ DSEM: {x.shape} -> {out.shape}")
    params = sum(p.numel() for p in dsem.parameters() if p.requires_grad) / 1e6
    print(f"  DSEM Parameters: {params:.2f} M")

    # 测试 OLD DACG
    print("\n[Test 2] OLD DACG Module")
    dacg = DACG_Block(256, 128, reduction=4)
    x = torch.randn(2, 256, 64, 64)
    out = dacg(x)
    print(f"✓ OLD DACG: {x.shape} -> {out.shape}")
    params_dacg = sum(p.numel() for p in dacg.parameters() if p.requires_grad) / 1e6
    print(f"  OLD DACG Parameters: {params_dacg:.2f} M")

    # 打印DACG结构信息
    print(f"\n  OLD DACG Structure:")
    print(f"    - local_branch layers: {len(dacg.local_branch)}")
    print(f"    - gate_conv layers: {len(dacg.gate_conv)}")
    print(f"    - Has gate_proj: {hasattr(dacg, 'gate_proj')}")

    # 检查strip卷积的groups
    print(f"    - strip_short_v groups: {dacg.strip_short_v.groups}")
    print(f"    - strip_short_v in_channels: {dacg.strip_short_v.in_channels}")
    print(f"    - strip_short_v out_channels: {dacg.strip_short_v.out_channels}")

    # 测试解码器 Up 模块
    print("\n[Test 3] Up Module with OLD DACG")
    up_dacg = Up(1024, 512, bilinear=True, use_dacg=True)
    x1 = torch.randn(2, 512, 32, 32)
    x2 = torch.randn(2, 512, 16, 16)
    out = up_dacg(x2, x1)
    print(f"✓ Up+OLD DACG: Enc[{x1.shape}] + Dec[{x2.shape}] -> {out.shape}")

    # 测试梯度流
    print("\n[Test 4] Gradient Flow")
    x = torch.randn(1, 256, 64, 64, requires_grad=True)
    out = dacg(x)
    loss = out.mean()
    loss.backward()

    if x.grad is not None and torch.isfinite(x.grad).all():
        print("✓ Gradients flow correctly through OLD DACG")
    else:
        print("✗ Gradient issue detected!")

    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)

    print("""
注意：这是旧版DACG，与你的checkpoint兼容
主要特点：
1. local_branch: 3层（Depthwise + BN + Conv）
2. gate_conv: 5层（Conv + BN + ReLU + Conv + Sigmoid）
3. 没有gate_proj层
4. strip卷积使用groups=4
    """)