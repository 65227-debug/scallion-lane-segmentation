"""
Topology-Guided Spatial Module (TGSM) - 完整实现
拓扑引导空间模块

使用方法:
    from tgsm import get_tgsm

    # 获取TGSM模块
    tgsm = get_tgsm(version='enhanced', in_channels=512)

    # 前向传播
    topo_feat, topo_conf = tgsm(bottleneck_feature)

版本选择:
    - 'basic': 基础版,快速验证
    - 'enhanced': 增强版,推荐使用 (骨架+连通性建模)
    - 'advanced': 高级版,极致性能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== Enhanced Version (推荐) ====================

class EnhancedTopologyConfidence(nn.Module):
    """
    增强版拓扑置信度估计

    三重建模:
    1. 能量稳定性 (全局一致性)
    2. 骨架响应 (局部中心性)
    3. 连通性检测 (结构连续性)
    """

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

        # 骨架检测核
        self.register_buffer('skeleton_kernel', torch.tensor([
            [1, 1, 1],
            [1, 10, 1],
            [1, 1, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3) / 18.0)

        # 连通性检测核
        self.register_buffer('connect_kernel', torch.tensor([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3) / 8.0)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            topo_conf: [B, 1, H, W]
        """
        # Component 1: 能量稳定性
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = (x - mean).pow(2).mean(dim=(2, 3), keepdim=True)
        energy = (x - mean).pow(2) / (var + self.eps)
        energy_conf = 1.0 / (1.0 + energy.mean(dim=1, keepdim=True))

        # Component 2: 骨架响应
        x_mean = x.mean(dim=1, keepdim=True)
        skeleton_resp = F.conv2d(
            x_mean,
            self.skeleton_kernel.to(x.device),
            padding=1
        )
        skeleton_conf = torch.sigmoid(skeleton_resp * 2.0)

        # Component 3: 连通性响应
        connect_resp = F.conv2d(
            x_mean,
            self.connect_kernel.to(x.device),
            padding=1
        )
        connect_conf = torch.sigmoid(connect_resp - 0.3)

        # 融合三种置信度
        topo_conf = energy_conf * skeleton_conf * connect_conf

        # 平滑处理
        topo_conf = F.avg_pool2d(topo_conf, 3, stride=1, padding=1)

        return topo_conf


class MultiScaleDirectionalAggregation(nn.Module):
    """
    多尺度方向聚合
    适应不同长度的车道线段
    """

    def __init__(self, channels):
        super().__init__()

        self.scales = [1, 2, 4]
        num_scales = len(self.scales)
        channels_per_scale = channels // num_scales

        # 处理不能整除的情况
        self.channels_per_scale = [channels_per_scale] * num_scales
        remainder = channels - (channels_per_scale * num_scales)
        for i in range(remainder):
            self.channels_per_scale[i] += 1

        self.proj_h = nn.ModuleList([
            nn.Conv2d(channels, self.channels_per_scale[i], 1, bias=False)
            for i in range(num_scales)
        ])

        self.proj_w = nn.ModuleList([
            nn.Conv2d(channels, self.channels_per_scale[i], 1, bias=False)
            for i in range(num_scales)
        ])

        # 输出通道数应该是所有尺度的通道数之和 * 2
        total_out_channels = sum(self.channels_per_scale) * 2

        self.fusion = nn.Sequential(
            nn.Conv2d(total_out_channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        feats_h, feats_w = [], []

        for i, scale in enumerate(self.scales):
            # H方向
            if scale > 1:
                x_h = F.avg_pool2d(x, (1, scale), stride=(1, scale))
            else:
                x_h = x

            feat_h = x_h.mean(dim=3, keepdim=True)
            feat_h = self.proj_h[i](feat_h)
            feat_h = F.interpolate(feat_h, size=(H, W), mode='bilinear', align_corners=False)
            feats_h.append(feat_h)

            # W方向
            if scale > 1:
                x_w = F.avg_pool2d(x, (scale, 1), stride=(scale, 1))
            else:
                x_w = x

            feat_w = x_w.mean(dim=2, keepdim=True)
            feat_w = self.proj_w[i](feat_w)
            feat_w = F.interpolate(feat_w, size=(H, W), mode='bilinear', align_corners=False)
            feats_w.append(feat_w)

        out = self.fusion(torch.cat([
            torch.cat(feats_h, dim=1),
            torch.cat(feats_w, dim=1)
        ], dim=1))

        return out


class EnhancedTGSM(nn.Module):
    """
    增强版TGSM (推荐)

    创新点:
    1. 多尺度方向聚合
    2. 三重拓扑建模
    3. 自适应残差连接
    """

    def __init__(self, in_channels):
        super().__init__()

        self.dir_agg = MultiScaleDirectionalAggregation(in_channels)
        self.topo_conf = EnhancedTopologyConfidence()

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        structural_feat = self.dir_agg(x)
        topo_conf = self.topo_conf(structural_feat)
        topo_feat = self.proj(structural_feat)

        alpha = torch.clamp(self.alpha, 0.0, 1.0)
        enhanced_feat = x + alpha * topo_feat

        return enhanced_feat, topo_conf


# ==================== Basic Version ====================

class BasicDirectionalAggregation(nn.Module):
    """基础版方向聚合"""

    def __init__(self, channels):
        super().__init__()
        self.proj_h = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.proj_w = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        feat_h = x.mean(dim=3, keepdim=True)
        feat_h = self.proj_h(feat_h)

        feat_w = x.mean(dim=2, keepdim=True)
        feat_w = self.proj_w(feat_w)

        out = feat_h + feat_w
        out = self.norm(out)
        out = self.act(out)

        return out


class BasicTopologyConfidence(nn.Module):
    """基础版拓扑置信度"""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = (x - mean).pow(2).mean(dim=(2, 3), keepdim=True)
        energy = (x - mean).pow(2) / (var + self.eps)
        topo_conf = 1.0 / (1.0 + energy.mean(dim=1, keepdim=True))

        return topo_conf


class BasicTGSM(nn.Module):
    """基础版TGSM"""

    def __init__(self, in_channels):
        super().__init__()

        self.dir_agg = BasicDirectionalAggregation(in_channels)
        self.topo_conf = BasicTopologyConfidence()

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        structural_feat = self.dir_agg(x)
        topo_conf = self.topo_conf(structural_feat)
        topo_feat = self.proj(structural_feat)

        return topo_feat, topo_conf


# ==================== Advanced Version ====================

class AdaptiveDirectionalAggregation(nn.Module):
    """自适应方向聚合"""

    def __init__(self, channels):
        super().__init__()

        self.dir_weight = nn.Parameter(torch.ones(2))

        self.scales = [1, 2, 4]
        num_scales = len(self.scales)
        channels_per_scale = channels // num_scales

        # 处理不能整除的情况
        self.channels_per_scale = [channels_per_scale] * num_scales
        remainder = channels - (channels_per_scale * num_scales)
        for i in range(remainder):
            self.channels_per_scale[i] += 1

        self.proj_h = nn.ModuleList([
            nn.Conv2d(channels, self.channels_per_scale[i], 1, bias=False)
            for i in range(num_scales)
        ])
        self.proj_w = nn.ModuleList([
            nn.Conv2d(channels, self.channels_per_scale[i], 1, bias=False)
            for i in range(num_scales)
        ])

        # 输出通道数应该是所有尺度的通道数之和 * 2
        total_out_channels = sum(self.channels_per_scale) * 2

        self.fusion = nn.Sequential(
            nn.Conv2d(total_out_channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        B, C, H, W = x.shape

        w_h, w_w = F.softmax(self.dir_weight, dim=0)

        feats_h, feats_w = [], []

        for i, scale in enumerate(self.scales):
            if scale > 1:
                x_h = F.avg_pool2d(x, (1, scale), stride=(1, scale))
            else:
                x_h = x
            feat_h = self.proj_h[i](x_h.mean(dim=3, keepdim=True))
            feat_h = F.interpolate(feat_h, size=(H, W), mode='bilinear', align_corners=False)
            feats_h.append(feat_h * w_h)

            if scale > 1:
                x_w = F.avg_pool2d(x, (scale, 1), stride=(scale, 1))
            else:
                x_w = x
            feat_w = self.proj_w[i](x_w.mean(dim=2, keepdim=True))
            feat_w = F.interpolate(feat_w, size=(H, W), mode='bilinear', align_corners=False)
            feats_w.append(feat_w * w_w)

        out = self.fusion(torch.cat([
            torch.cat(feats_h, dim=1),
            torch.cat(feats_w, dim=1)
        ], dim=1))

        return out


class AdvancedTGSM(nn.Module):
    """高级版TGSM"""

    def __init__(self, in_channels):
        super().__init__()

        self.dir_agg = AdaptiveDirectionalAggregation(in_channels)
        self.topo_conf = EnhancedTopologyConfidence()

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.topo_gate = nn.Sequential(
            nn.Conv2d(1, in_channels, 1),
            nn.Sigmoid()
        )

        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.3))

    def forward(self, x):
        structural_feat = self.dir_agg(x)
        topo_conf = self.topo_conf(structural_feat)

        topo_gate = self.topo_gate(topo_conf)
        topo_feat = self.proj(structural_feat) * topo_gate

        alpha = torch.clamp(self.alpha, 0.0, 1.0)
        beta = torch.clamp(self.beta, 0.0, 1.0)

        enhanced_feat = alpha * x + beta * topo_feat

        return enhanced_feat, topo_conf


# ==================== Factory Function ====================

def get_tgsm(version='enhanced', in_channels=512):
    """
    获取TGSM模块

    Args:
        version: 'basic', 'enhanced', 'advanced'
        in_channels: 输入通道数

    Returns:
        TGSM模块
    """
    if version == 'basic':
        return BasicTGSM(in_channels)
    elif version == 'enhanced':
        return EnhancedTGSM(in_channels)
    elif version == 'advanced':
        return AdvancedTGSM(in_channels)
    else:
        raise ValueError(f"Unknown TGSM version: {version}")