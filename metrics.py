"""
Evaluation metrics for semantic segmentation
包括 MIoU, MPA, FLOPs, FPS, Parameters
"""

import torch
import torch.nn as nn
import numpy as np
import time
from thop import profile, clever_format


def calculate_miou(pred, target, num_classes):
    """
    计算 Mean Intersection over Union (MIoU)

    Args:
        pred: 预测结果 (B, H, W) 或 (B, C, H, W)
        target: 真实标签 (B, H, W)
        num_classes: 类别数量

    Returns:
        miou: Mean IoU值 (百分比)
    """
    if pred.dim() == 4:
        pred = pred.argmax(dim=1)

    pred = pred.cpu().numpy()
    target = target.cpu().numpy()

    iou_list = []
    for cls in range(num_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)

        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()

        if union == 0:
            iou_list.append(float('nan'))
        else:
            iou_list.append(intersection / union)

    # 忽略 NaN 值计算平均
    iou_array = np.array(iou_list)
    miou = np.nanmean(iou_array) * 100

    return miou


def calculate_mpa(pred, target, num_classes):
    """
    计算 Mean Pixel Accuracy (MPA)

    Args:
        pred: 预测结果 (B, H, W) 或 (B, C, H, W)
        target: 真实标签 (B, H, W)
        num_classes: 类别数量

    Returns:
        mpa: Mean Pixel Accuracy值 (百分比)
    """
    if pred.dim() == 4:
        pred = pred.argmax(dim=1)

    pred = pred.cpu().numpy()
    target = target.cpu().numpy()

    acc_list = []
    for cls in range(num_classes):
        target_mask = (target == cls)

        if target_mask.sum() == 0:
            acc_list.append(float('nan'))
        else:
            pred_mask = (pred == cls)
            correct = np.logical_and(pred_mask, target_mask).sum()
            acc_list.append(correct / target_mask.sum())

    # 忽略 NaN 值计算平均
    acc_array = np.array(acc_list)
    mpa = np.nanmean(acc_array) * 100

    return mpa


def calculate_flops(model, input_size=(1, 3, 540, 960), device='cuda'):
    """
    计算模型的 FLOPs

    Args:
        model: 神经网络模型
        input_size: 输入张量大小 (B, C, H, W)
        device: 设备

    Returns:
        flops: FLOPs值 (G)
    """
    model.eval()
    input_tensor = torch.randn(input_size).to(device)

    try:
        flops, params = profile(model, inputs=(input_tensor,), verbose=False)
        flops_g = flops / 1e9  # 转换为 GFLOPs
        return flops_g
    except Exception as e:
        print(f"FLOPs calculation failed: {e}")
        return 0.0


def calculate_fps(model, input_size=(1, 3, 540, 960), device='cuda', num_iterations=100):
    """
    计算模型的推理速度 FPS

    Args:
        model: 神经网络模型
        input_size: 输入张量大小 (B, C, H, W)
        device: 设备
        num_iterations: 测试迭代次数

    Returns:
        fps: 每秒帧数
    """
    model.eval()
    input_tensor = torch.randn(input_size).to(device)

    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)

    # 同步 CUDA
    if device == 'cuda':
        torch.cuda.synchronize()

    # 计时
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(input_tensor)

    if device == 'cuda':
        torch.cuda.synchronize()

    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = num_iterations / elapsed_time

    return fps


def count_parameters(model):
    """
    计算模型参数量

    Args:
        model: 神经网络模型

    Returns:
        total_params: 总参数量 (M)
        trainable_params: 可训练参数量 (M)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    total_params_m = total_params / 1e6
    trainable_params_m = trainable_params / 1e6

    return total_params_m, trainable_params_m


def get_model_metrics(model, input_size=(1, 3, 512, 512), device='cuda'):
    """
    获取模型的所有静态指标

    Args:
        model: 神经网络模型
        input_size: 输入张量大小
        device: 设备

    Returns:
        metrics: 包含所有指标的字典
    """
    total_params, trainable_params = count_parameters(model)
    flops = calculate_flops(model, input_size, device)
    fps = calculate_fps(model, input_size, device)

    metrics = {
        'Parameters (M)': total_params,
        'Trainable Parameters (M)': trainable_params,
        'FLOPs (G)': flops,
        'FPS': fps
    }

    return metrics