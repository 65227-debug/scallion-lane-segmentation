"""
完整推理脚本 - 一体化版本
集成 UNet + DSEM + DACG + TGSM 模型架构和推理功能
支持单张图像和批量推理

使用方法:
    # 单张图像推理
    python inference_complete.py --image path/to/image.jpg --model path/to/model.pth --output result.png

    # 批量推理（带评估）
    python inference_complete.py --image-dir path/to/images --mask-dir path/to/masks --model path/to/model.pth --output-dir results

    # 指定模型配置
    python inference_complete.py --image test.jpg --model model.pth --use-dsem --use-dacg --use-tgsm --tgsm-version enhanced
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ==================== UNet 基础模块 ====================

class DoubleConv(nn.Module):
    """双卷积块 - 与 unet_parts.py 完全一致（conv1 + conv2 命名）"""

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

        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None

        if use_dsem:
            self.conv2 = nn.Sequential(
                nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                DSEM_Module(out_channels, out_channels, reduction=reduction)
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
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, use_dsem=False, reduction=4):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, use_dsem=use_dsem, reduction=reduction)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# ==================== DSEM 模块 ====================

class DSEM_Module(nn.Module):
    """
    Directional Structure Encoding Module (DSEM)
    与 unet_parts.py 中的 DSEM 完全一致
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

        # 方向条形卷积
        self.dir_convs = nn.ModuleList()
        for k in kernel_sizes:
            self.dir_convs.append(
                nn.Conv2d(in_channels, inter_channels,
                          kernel_size=(k, 1), padding=(k // 2, 0), bias=False)
            )
            self.dir_convs.append(
                nn.Conv2d(in_channels, inter_channels,
                          kernel_size=(1, k), padding=(0, k // 2), bias=False)
            )

        num_groups = min(32, inter_channels * len(self.dir_convs))
        while (inter_channels * len(self.dir_convs)) % num_groups != 0:
            num_groups -= 1

        if norm == 'group':
            self.norm = nn.GroupNorm(num_groups, inter_channels * len(self.dir_convs))
        else:
            self.norm = nn.BatchNorm2d(inter_channels * len(self.dir_convs))

        self.fusion = nn.Sequential(
            nn.Conv2d(inter_channels * len(self.dir_convs),
                      out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()

        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        identity = self.residual(x)

        dir_feats = [conv(x) for conv in self.dir_convs]
        dir_feat = torch.cat(dir_feats, dim=1)
        dir_feat = self.norm(dir_feat)

        structure_feat = self.fusion(dir_feat)

        alpha = torch.clamp(self.alpha, 0.0, 1.0)
        out = identity + alpha * structure_feat

        return out


# ==================== DACG 模块 ====================

class DACG_Block(nn.Module):
    """
    Direction-Aware Context Gating Block for Decoder
    与 unet_parts.py 中的 DACG_Block 完全一致
    """

    def __init__(self, in_channels: int, out_channels: int, reduction: int = 4):
        super().__init__()

        # 1. 输入投影
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 2. 局部特征分支（Depthwise + 1x1）
        self.local_branch = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,
                      groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
        )

        # 3. 多尺度条形卷积（vertical接受4通道，horizontal接受1通道）
        self.strip_short_v = nn.Conv2d(4, out_channels, kernel_size=(3, 1),
                                       padding=(1, 0), bias=False)
        self.strip_short_h = nn.Conv2d(1, out_channels, kernel_size=(1, 3),
                                       padding=(0, 1), bias=False)
        self.bn_short = nn.BatchNorm2d(out_channels)

        self.strip_mid_v = nn.Conv2d(4, out_channels, kernel_size=(7, 1),
                                     padding=(3, 0), bias=False)
        self.strip_mid_h = nn.Conv2d(1, out_channels, kernel_size=(1, 7),
                                     padding=(0, 3), bias=False)
        self.bn_mid = nn.BatchNorm2d(out_channels)

        self.strip_long_v = nn.Conv2d(4, out_channels, kernel_size=(11, 1),
                                      padding=(5, 0), bias=False)
        self.strip_long_h = nn.Conv2d(1, out_channels, kernel_size=(1, 11),
                                      padding=(0, 5), bias=False)
        self.bn_long = nn.BatchNorm2d(out_channels)

        # 4. 门控调制路径（5层）
        gate_input_channels = out_channels * 3
        self.gate_conv = nn.Sequential(
            nn.Conv2d(gate_input_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True),
            nn.Sigmoid()
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

        # 8. 可学习融合权重
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)
        self.beta = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x):
        residual = self.shortcut(x)
        x_proj = self.input_proj(x)
        x_local = self.local_branch(x_proj)

        B, C, H, W = x_proj.shape
        split_size = C // 4
        x_reshaped = x_proj.view(B, 4, split_size, H, W)
        x_strip_4ch = x_reshaped.mean(dim=2)

        feat_short_v = self.strip_short_v(x_strip_4ch)
        feat_short_1ch = feat_short_v.mean(dim=1, keepdim=True)
        feat_short = self.strip_short_h(feat_short_1ch)
        feat_short = self.bn_short(feat_short)

        feat_mid_v = self.strip_mid_v(x_strip_4ch)
        feat_mid_1ch = feat_mid_v.mean(dim=1, keepdim=True)
        feat_mid = self.strip_mid_h(feat_mid_1ch)
        feat_mid = self.bn_mid(feat_mid)

        feat_long_v = self.strip_long_v(x_strip_4ch)
        feat_long_1ch = feat_long_v.mean(dim=1, keepdim=True)
        feat_long = self.strip_long_h(feat_long_1ch)
        feat_long = self.bn_long(feat_long)

        multi_scale_feat = torch.cat([feat_short, feat_mid, feat_long], dim=1)

        gate = self.gate_conv(multi_scale_feat)
        x_gated = x_local * gate

        x_add = self.add_path(multi_scale_feat)

        x_fused = torch.cat([x_gated, x_add], dim=1)
        x_out = self.fusion(x_fused)

        alpha = torch.clamp(self.alpha, 0.0, 1.0)
        beta = torch.clamp(self.beta, 0.0, 1.0)
        out = alpha * x_out + beta * residual

        return out


# ==================== TGSM 模块 ====================

class EnhancedTopologyConfidence(nn.Module):
    """增强版拓扑置信度估计"""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

        self.register_buffer('skeleton_kernel', torch.tensor([
            [1, 1, 1],
            [1, 10, 1],
            [1, 1, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3) / 18.0)

        self.register_buffer('connect_kernel', torch.tensor([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3) / 8.0)

    def forward(self, x):
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = (x - mean).pow(2).mean(dim=(2, 3), keepdim=True)
        energy = (x - mean).pow(2) / (var + self.eps)
        energy_conf = 1.0 / (1.0 + energy.mean(dim=1, keepdim=True))

        x_mean = x.mean(dim=1, keepdim=True)
        skeleton_resp = F.conv2d(x_mean, self.skeleton_kernel.to(x.device), padding=1)
        skeleton_conf = torch.sigmoid(skeleton_resp * 2.0)

        connect_resp = F.conv2d(x_mean, self.connect_kernel.to(x.device), padding=1)
        connect_conf = torch.sigmoid(connect_resp - 0.3)

        topo_conf = energy_conf * skeleton_conf * connect_conf
        topo_conf = F.avg_pool2d(topo_conf, 3, stride=1, padding=1)

        return topo_conf


class MultiScaleDirectionalAggregation(nn.Module):
    """多尺度方向聚合"""

    def __init__(self, channels):
        super().__init__()

        self.scales = [1, 2, 4]
        num_scales = len(self.scales)
        channels_per_scale = channels // num_scales

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
            if scale > 1:
                x_h = F.avg_pool2d(x, (1, scale), stride=(1, scale))
            else:
                x_h = x

            feat_h = x_h.mean(dim=3, keepdim=True)
            feat_h = self.proj_h[i](feat_h)
            feat_h = F.interpolate(feat_h, size=(H, W), mode='bilinear', align_corners=False)
            feats_h.append(feat_h)

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
    """增强版TGSM"""

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


# ==================== UNet 上采样模块 ====================

class Up_with_TGSM(nn.Module):
    """上采样块 - 支持TGSM的拓扑置信度引导"""

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

    def forward(self, x1, x2, topo_conf=None):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# ==================== 完整 UNet 模型 ====================

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False,
                 use_dsem=False, reduction=4,
                 use_dacg=False,
                 use_tgsm=False, tgsm_version='enhanced'):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_dsem = use_dsem
        self.reduction = reduction
        self.use_dacg = use_dacg
        self.use_tgsm = use_tgsm
        self.tgsm_version = tgsm_version

        # 编码器
        self.inc = DoubleConv(n_channels, 64, use_dsem=use_dsem, reduction=reduction)
        self.down1 = Down(64, 128, use_dsem=use_dsem, reduction=reduction)
        self.down2 = Down(128, 256, use_dsem=use_dsem, reduction=reduction)
        self.down3 = Down(256, 512, use_dsem=use_dsem, reduction=reduction)

        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, use_dsem=use_dsem, reduction=reduction)

        # TGSM (Bottleneck)
        if use_tgsm:
            self.tgsm = EnhancedTGSM(1024 // factor)

        # 解码器
        self.up1 = Up_with_TGSM(1024, 512 // factor, bilinear, use_dacg=use_dacg)
        self.up2 = Up_with_TGSM(512, 256 // factor, bilinear, use_dacg=use_dacg)
        self.up3 = Up_with_TGSM(256, 128 // factor, bilinear, use_dacg=use_dacg)
        self.up4 = Up_with_TGSM(128, 64, bilinear, use_dacg=use_dacg)

        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        if self.use_tgsm:
            topo_feat, topo_conf = self.tgsm(x5)
            x5 = topo_feat
        else:
            topo_conf = None

        x = self.up1(x5, x4, topo_conf)
        x = self.up2(x, x3, topo_conf)
        x = self.up3(x, x2, topo_conf)
        x = self.up4(x, x1, topo_conf)

        logits = self.outc(x)
        return logits


# ==================== 模型加载工具 ====================

def clean_state_dict(state_dict):
    """清理state_dict，移除额外的统计信息"""
    keys_to_remove = [k for k in state_dict.keys() if 'total_ops' in k or 'total_params' in k]

    unwanted_keys = ['mask_values', 'bridge.']
    for key in list(state_dict.keys()):
        if any(unwanted in key for unwanted in unwanted_keys):
            keys_to_remove.append(key)

    for key in keys_to_remove:
        if key in state_dict:
            del state_dict[key]

    return state_dict


def load_model_flexible(model, checkpoint_path, device):
    """灵活加载模型权重，自动处理不匹配"""
    logging.info(f'Loading checkpoint from {checkpoint_path}')
    state_dict = torch.load(checkpoint_path, map_location=device)

    state_dict = clean_state_dict(state_dict)

    # 键名映射：tgb -> tgsm
    tgb_count = 0
    new_state_dict_renamed = {}
    for key, value in state_dict.items():
        if key.startswith('tgb.'):
            new_key = key.replace('tgb.', 'tgsm.')
            new_state_dict_renamed[new_key] = value
            tgb_count += 1
        else:
            new_state_dict_renamed[key] = value

    if tgb_count > 0:
        logging.info(f'Renamed {tgb_count} keys from tgb.* to tgsm.*')
        state_dict = new_state_dict_renamed

    model_state = model.state_dict()

    matched_keys = []
    unmatched_keys = []
    missing_keys = []

    new_state_dict = {}
    for key, value in state_dict.items():
        if key in model_state:
            if value.shape == model_state[key].shape:
                new_state_dict[key] = value
                matched_keys.append(key)
            else:
                unmatched_keys.append(f"{key}: {value.shape} -> {model_state[key].shape}")
        else:
            unmatched_keys.append(f"{key}: not in model")

    for key in model_state.keys():
        if key not in state_dict:
            missing_keys.append(key)

    model.load_state_dict(new_state_dict, strict=False)

    logging.info(f'Successfully loaded {len(matched_keys)}/{len(state_dict)} parameters from checkpoint')

    if len(missing_keys) > 0:
        tgsm_keys = [k for k in missing_keys if 'tgsm' in k.lower()]
        other_keys = [k for k in missing_keys if 'tgsm' not in k.lower()]

        if len(tgsm_keys) > 0:
            logging.warning(
                f'TGSM module not found in checkpoint ({len(tgsm_keys)} parameters will be randomly initialized)')

        if len(other_keys) > 0:
            logging.warning(f'{len(other_keys)} other parameters missing in checkpoint')

    return model


# ==================== 图像处理工具 ====================

def load_image(image_path, scale=0.5):
    """加载并预处理图像"""
    img = Image.open(image_path)
    w, h = img.size
    newW, newH = int(scale * w), int(scale * h)

    img = img.resize((newW, newH), resample=Image.BICUBIC)
    img = np.asarray(img)

    if img.ndim == 2:
        img = img[np.newaxis, ...]
    else:
        img = img.transpose((2, 0, 1))

    if (img > 1).any():
        img = img / 255.0

    return torch.from_numpy(img).float()


def load_mask(mask_path, scale=0.5, label_mapping=None):
    """加载标签图像"""
    mask = Image.open(mask_path)
    w, h = mask.size
    newW, newH = int(scale * w), int(scale * h)

    mask = mask.resize((newW, newH), resample=Image.NEAREST)
    mask = np.asarray(mask)

    if label_mapping is not None:
        mapped_mask = np.ones_like(mask) * 255
        for old_label, new_label in label_mapping.items():
            mapped_mask[mask == old_label] = new_label
        mask = mapped_mask

    return mask


def predict_image(model, image_tensor, device):
    """对单张图像进行推理"""
    model.eval()
    image_tensor = image_tensor.unsqueeze(0).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = model(image_tensor)
        pred = output.argmax(dim=1)[0].cpu().numpy()

    return pred


def save_prediction(pred, save_path, original_size=None, label_map=None):
    """保存预测结果"""
    if label_map is not None:
        pred_original = np.zeros_like(pred, dtype=np.uint8)
        for new_label, old_label in label_map.items():
            pred_original[pred == new_label] = old_label
        pred = pred_original

    pred_img = Image.fromarray(pred.astype(np.uint8))

    if original_size is not None:
        pred_img = pred_img.resize(original_size, resample=Image.NEAREST)

    pred_img.save(save_path)


# ==================== 评估工具 ====================

def calculate_iou_per_image(pred, target, num_classes=3, ignore_index=255):
    """计算单张图像的IoU"""
    valid_mask = (target != ignore_index)

    ious = {}
    class_names = ['Background', 'Left_Lane', 'Right_Lane']

    iou_list = []
    for c in range(num_classes):
        pred_c = (pred == c) & valid_mask
        target_c = (target == c) & valid_mask

        intersection = (pred_c & target_c).sum()
        union = (pred_c | target_c).sum()

        if union > 0:
            iou = intersection / union
        else:
            iou = 0.0

        class_name = class_names[c] if c < len(class_names) else f'Class_{c}'
        ious[f'IoU_{class_name}'] = iou
        iou_list.append(iou)

    valid_ious = [iou for iou in iou_list if iou > 0]
    ious['mIoU'] = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0

    lane_ious = iou_list[1:]
    valid_lane_ious = [iou for iou in lane_ious if iou > 0]
    ious['mIoU_Lane'] = sum(valid_lane_ious) / len(valid_lane_ious) if valid_lane_ious else 0.0

    return ious


def visualize_comparison(image_path, mask_path, pred, save_path, iou_scores):
    """可视化对比图"""
    img = np.array(Image.open(image_path))
    mask = np.array(Image.open(mask_path))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 原图
    axes[0].imshow(img)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # 真实标签
    mask_color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    mask_color[mask == 0] = [0, 0, 0]
    mask_color[mask == 85] = [255, 0, 0]
    mask_color[mask == 170] = [0, 0, 255]

    axes[1].imshow(mask_color)
    axes[1].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    # 预测结果
    pred_color = np.zeros((*pred.shape, 3), dtype=np.uint8)
    pred_color[pred == 0] = [0, 0, 0]
    pred_color[pred == 1] = [255, 0, 0]
    pred_color[pred == 2] = [0, 0, 255]

    axes[2].imshow(pred_color)
    title = f'Prediction\n'
    title += f'mIoU: {iou_scores["mIoU"] * 100:.2f}% | '
    title += f'mIoU_Lane: {iou_scores["mIoU_Lane"] * 100:.2f}%\n'
    title += f'BG: {iou_scores["IoU_Background"] * 100:.1f}% | '
    title += f'Left: {iou_scores["IoU_Left_Lane"] * 100:.1f}% | '
    title += f'Right: {iou_scores["IoU_Right_Lane"] * 100:.1f}%'
    axes[2].set_title(title, fontsize=12, fontweight='bold')
    axes[2].axis('off')

    # 图例
    legend_elements = [
        mpatches.Patch(facecolor='black', label='Background'),
        mpatches.Patch(facecolor='red', label='Left Lane'),
        mpatches.Patch(facecolor='blue', label='Right Lane')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description='Lane Segmentation Inference')

    # 基本参数
    parser.add_argument('--image', type=str, help='单张图像路径')
    parser.add_argument('--image-dir', type=str, help='图像目录（批量推理）')
    parser.add_argument('--mask-dir', type=str, help='标签目录（用于评估）')
    parser.add_argument('--model', '--model-path', type=str, required=True, help='模型权重路径')
    parser.add_argument('--output', type=str, help='输出文件路径（单张图像）')
    parser.add_argument('--output-dir', type=str, default='./inference_results', help='输出目录（批量推理）')

    # 模型参数
    parser.add_argument('--classes', type=int, default=3, help='类别数')
    parser.add_argument('--scale', type=float, default=0.5, help='图像缩放比例')
    parser.add_argument('--use-dsem', action='store_true', help='使用DSEM模块')
    parser.add_argument('--use-dacg', action='store_true', help='使用DACG模块')
    parser.add_argument('--use-tgsm', action='store_true', help='使用TGSM模块')
    parser.add_argument('--tgsm-version', type=str, default='enhanced', choices=['basic', 'enhanced', 'advanced'])
    parser.add_argument('--bilinear', action='store_true', help='使用双线性上采样')

    # 评估参数
    parser.add_argument('--num-worst', type=int, default=20, help='保存最差的N张图像')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # 检查输入
    if args.image is None and args.image_dir is None:
        parser.error('必须指定 --image 或 --image-dir')

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # 构建模型配置信息
    model_config = []
    if args.use_dsem:
        model_config.append("DSEM")
    if args.use_dacg:
        model_config.append("DACG")
    if args.use_tgsm:
        model_config.append(f"TGSM-{args.tgsm_version}")

    config_str = " + ".join(model_config) if model_config else "Baseline"
    logging.info(f'Model configuration: {config_str}')

    # 加载模型
    model = UNet(
        n_channels=3,
        n_classes=args.classes,
        bilinear=args.bilinear,
        use_dsem=args.use_dsem,
        use_dacg=args.use_dacg,
        use_tgsm=args.use_tgsm,
        tgsm_version=args.tgsm_version
    )

    model = load_model_flexible(model, args.model, device)
    model.to(device=device)
    logging.info('Model loaded successfully')

    # 单张图像推理
    if args.image:
        logging.info(f'Processing single image: {args.image}')

        image = load_image(args.image, scale=args.scale)
        pred = predict_image(model, image, device)

        output_path = args.output if args.output else 'prediction.png'

        # 获取原始图像尺寸
        original_img = Image.open(args.image)
        save_prediction(pred, output_path, original_size=original_img.size,
                        label_map={0: 0, 1: 85, 2: 170})

        logging.info(f'Prediction saved to {output_path}')

    # 批量推理
    elif args.image_dir:
        output_dir = Path(args.output_dir)
        pred_dir = output_dir / 'predictions'
        worst_dir = output_dir / 'worst_cases'
        pred_dir.mkdir(parents=True, exist_ok=True)
        worst_dir.mkdir(parents=True, exist_ok=True)

        image_dir = Path(args.image_dir)
        image_files = sorted(list(image_dir.glob('*.jpg')) +
                             list(image_dir.glob('*.png')) +
                             list(image_dir.glob('*.jpeg')))

        if len(image_files) == 0:
            logging.error(f'No images found in {image_dir}')
            return

        logging.info(f'Found {len(image_files)} images')

        label_mapping = {0: 0, 85: 1, 170: 2}
        results = []

        # 推理所有图像
        logging.info('Starting inference...')
        for img_path in tqdm(image_files, desc='Processing images'):
            img_name = img_path.stem

            # 查找对应的标签文件
            mask_path = None
            if args.mask_dir:
                mask_dir = Path(args.mask_dir)
                for ext in ['.png', '.jpg', '.jpeg']:
                    potential_mask = mask_dir / f"{img_name}{ext}"
                    if potential_mask.exists():
                        mask_path = potential_mask
                        break

            try:
                image = load_image(img_path, scale=args.scale)
                pred = predict_image(model, image, device)

                # 保存预测结果
                original_img = Image.open(img_path)
                save_prediction(pred, pred_dir / f"{img_name}.png",
                                original_size=original_img.size,
                                label_map={0: 0, 1: 85, 2: 170})

                # 如果有标签，计算IoU
                if mask_path is not None:
                    mask_gt = load_mask(mask_path, scale=args.scale, label_mapping=label_mapping)
                    iou_scores = calculate_iou_per_image(pred, mask_gt, num_classes=args.classes)

                    results.append({
                        'image_name': img_name,
                        'image_path': img_path,
                        'mask_path': mask_path,
                        'pred': pred,
                        'iou_scores': iou_scores,
                        'mIoU': iou_scores['mIoU'],
                        'mIoU_Lane': iou_scores['mIoU_Lane']
                    })

            except Exception as e:
                logging.error(f'Error processing {img_name}: {e}')
                continue

        logging.info(f'Inference completed. Predictions saved to {pred_dir}')

        # 评估结果
        if len(results) > 0:
            avg_miou = np.mean([r['mIoU'] for r in results])
            avg_miou_lane = np.mean([r['mIoU_Lane'] for r in results])

            logging.info('\n' + '=' * 60)
            logging.info('Overall Statistics:')
            logging.info('=' * 60)
            logging.info(f'Model: {config_str}')
            logging.info(f'Total images processed: {len(results)}')
            logging.info(f'Average mIoU: {avg_miou * 100:.2f}%')
            logging.info(f'Average mIoU_Lane: {avg_miou_lane * 100:.2f}%')

            # 保存最差案例
            results_sorted = sorted(results, key=lambda x: x['mIoU_Lane'])
            worst_cases = results_sorted[:args.num_worst]

            logging.info('\n' + '=' * 60)
            logging.info(f'Top {args.num_worst} Worst Cases (by mIoU_Lane):')
            logging.info('=' * 60)

            for i, case in enumerate(worst_cases, 1):
                info = f"[{i}] {case['image_name']}\n"
                info += f"    mIoU: {case['mIoU'] * 100:.2f}%\n"
                info += f"    mIoU_Lane: {case['mIoU_Lane'] * 100:.2f}%\n"
                logging.info(info)

                vis_path = worst_dir / f'worst_{i:03d}_{case["image_name"]}.png'
                visualize_comparison(
                    case['image_path'],
                    case['mask_path'],
                    case['pred'],
                    vis_path,
                    case['iou_scores']
                )

            logging.info(f'\nWorst cases visualization saved to {worst_dir}')

            # 保存CSV
            import csv
            csv_path = output_dir / 'all_results.csv'
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Image Name', 'mIoU (%)', 'mIoU_Lane (%)',
                                 'IoU_Background (%)', 'IoU_Left_Lane (%)', 'IoU_Right_Lane (%)'])

                for r in sorted(results, key=lambda x: x['mIoU_Lane']):
                    writer.writerow([
                        r['image_name'],
                        f"{r['mIoU'] * 100:.2f}",
                        f"{r['mIoU_Lane'] * 100:.2f}",
                        f"{r['iou_scores']['IoU_Background'] * 100:.2f}",
                        f"{r['iou_scores']['IoU_Left_Lane'] * 100:.2f}",
                        f"{r['iou_scores']['IoU_Right_Lane'] * 100:.2f}"
                    ])

            logging.info(f'All results saved to {csv_path}')

        logging.info('\nInference completed!')


if __name__ == '__main__':
    main()