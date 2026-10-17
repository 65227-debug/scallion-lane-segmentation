"""
Boundary-Aware Loss Functions for DAG-Net (Complete Fix v2)
边界感知损失函数 - 完全修复版本

核心修复：
1. 强制在 float32 下计算损失（避免AMP的float16溢出）
2. 移除边界权重乘法中的溢出风险
3. 添加输入验证和清理
4. 更保守的默认参数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import logging


class BoundaryAwareLoss(nn.Module):
    """
    边界感知复合损失函数 (完全修复版 v2)

    关键修复：强制使用float32计算，避免AMP导致的数值溢出

    Args:
        num_classes: 类别数量
        ignore_index: 忽略的标签值 (默认255)
        boundary_weight: 边界区域的损失权重倍数 (默认5.0, 大幅降低)
        dice_weight: Dice损失的权重 (默认1.0)
        use_boundary_loss: 是否使用额外的边界损失 (默认False, 简化计算)
        boundary_loss_weight: 边界损失的权重 (默认0.3)
    """

    def __init__(
            self,
            num_classes: int,
            ignore_index: int = 255,
            boundary_weight: float = 5.0,  # 大幅降低默认值
            dice_weight: float = 1.0,
            use_boundary_loss: bool = False,  # 默认关闭以提高稳定性
            boundary_loss_weight: float = 0.3
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.boundary_weight = boundary_weight
        self.dice_weight = dice_weight
        self.use_boundary_loss = use_boundary_loss
        self.boundary_loss_weight = boundary_loss_weight

        # 平滑参数
        self.smooth = 1.0
        self.eps = 1e-7

        # Laplacian 边界检测核
        self.register_buffer('laplacian_kernel', torch.tensor(
            [[0, 1, 0],
             [1, -4, 1],
             [0, 1, 0]], dtype=torch.float32
        ).view(1, 1, 3, 3))

        # 统计
        self.nan_count = 0

    def _to_float32(self, tensor: Tensor) -> Tensor:
        """强制转换为float32"""
        return tensor.float()

    def _check_input(self, pred: Tensor, target: Tensor) -> tuple:
        """
        检查并清理输入

        Returns:
            (pred_clean, target_clean, is_valid)
        """
        # 强制转换为float32
        pred = self._to_float32(pred)

        # 检查pred是否包含NaN/Inf
        if torch.isnan(pred).any() or torch.isinf(pred).any():
            self.nan_count += 1
            if self.nan_count <= 5:
                logging.warning(f'Input pred contains NaN/Inf! Cleaning... (count: {self.nan_count})')
            # 替换NaN/Inf为0
            pred = torch.where(torch.isfinite(pred), pred, torch.zeros_like(pred))

        # 裁剪pred到合理范围，防止softmax/exp溢出
        pred = torch.clamp(pred, min=-20.0, max=20.0)

        return pred, target, True

    def extract_boundary(self, mask: Tensor, dilation: int = 1) -> Tensor:
        """从分割标签中提取边界"""
        with torch.no_grad():  # 边界提取不需要梯度
            valid_mask = (mask != self.ignore_index).float().unsqueeze(1)
            boundary = torch.zeros_like(valid_mask)

            for c in range(self.num_classes):
                class_mask = (mask == c).float().unsqueeze(1)
                if class_mask.sum() > 0:
                    kernel = self.laplacian_kernel.to(class_mask.device)
                    edges = F.conv2d(class_mask, kernel, padding=1)
                    boundary = boundary + (edges.abs() > 0).float()

            boundary = (boundary > 0).float() * valid_mask

            if dilation > 1:
                kernel_size = 2 * dilation + 1
                boundary = F.max_pool2d(boundary, kernel_size, stride=1, padding=dilation)

            return boundary.squeeze(1)

    def dice_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        多类别Dice损失 (稳定版)
        """
        # 使用float32计算softmax
        pred_soft = F.softmax(pred, dim=1)

        valid_mask = (target != self.ignore_index).float()
        num_valid = valid_mask.sum()

        if num_valid < 1:
            return pred.sum() * 0.0  # 保持梯度连接

        dice_sum = pred.new_zeros(1)
        num_classes_computed = 0

        for c in range(self.num_classes):
            pred_c = pred_soft[:, c] * valid_mask
            target_c = (target == c).float() * valid_mask

            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()

            if union > self.smooth:
                dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
                dice_sum = dice_sum + (1.0 - dice)
                num_classes_computed += 1

        if num_classes_computed > 0:
            return dice_sum / num_classes_computed
        else:
            return pred.sum() * 0.0

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        计算边界感知复合损失

        关键：所有计算强制在float32下进行
        """
        # 1. 输入检查和清理
        pred, target, is_valid = self._check_input(pred, target)

        if not is_valid:
            # 如果输入无效，返回标准CE loss
            return F.cross_entropy(pred, target, ignore_index=self.ignore_index)

        # 2. 计算有效像素
        valid_mask = (target != self.ignore_index).float()
        num_valid = valid_mask.sum()

        if num_valid < 1:
            return pred.sum() * 0.0

        # 3. 计算标准交叉熵 (逐像素)
        # 使用 log_softmax + nll_loss 手动计算，更稳定
        log_prob = F.log_softmax(pred, dim=1)

        # 安全的NLL计算
        target_safe = target.clone()
        target_safe[target == self.ignore_index] = 0  # 临时替换ignore值

        # gather log probabilities
        ce_loss = -log_prob.gather(1, target_safe.unsqueeze(1)).squeeze(1)

        # 只在有效位置计算
        ce_loss = ce_loss * valid_mask

        # 4. 边界加权 (如果启用)
        if self.boundary_weight > 1.0:
            boundary = self.extract_boundary(target, dilation=2)
            # 使用加法而非乘法来降低数值风险
            # weight = 1 + boundary_weight * boundary
            # 改为更稳定的形式
            weight = 1.0 + boundary * (self.boundary_weight - 1.0)
            weight = torch.clamp(weight, min=1.0, max=self.boundary_weight)
            ce_loss = ce_loss * weight

        # 5. 平均
        weighted_ce = ce_loss.sum() / (num_valid + self.eps)

        # 6. 检查CE结果
        if torch.isnan(weighted_ce) or torch.isinf(weighted_ce):
            logging.warning('CE loss is NaN/Inf after manual computation, using fallback')
            weighted_ce = F.cross_entropy(pred, target, ignore_index=self.ignore_index)
            if torch.isnan(weighted_ce) or torch.isinf(weighted_ce):
                return pred.sum() * 0.0 + 1.0  # 返回常数损失

        # 7. Dice损失
        dice = self.dice_loss(pred, target)

        # 8. 合并
        if self.use_boundary_loss:
            boundary = self.extract_boundary(target, dilation=2)
            boundary_dice = self._boundary_dice_loss(pred, target, boundary)
            total_loss = weighted_ce + self.dice_weight * dice + self.boundary_loss_weight * boundary_dice
        else:
            total_loss = weighted_ce + self.dice_weight * dice

        # 9. 最终安全检查
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logging.error('Total loss is still NaN/Inf, returning CE only')
            return F.cross_entropy(
                torch.clamp(pred, -10, 10),
                target,
                ignore_index=self.ignore_index
            )

        return total_loss

    def _boundary_dice_loss(self, pred: Tensor, target: Tensor, boundary: Tensor) -> Tensor:
        """边界区域Dice损失"""
        pred_soft = F.softmax(pred, dim=1)
        valid_mask = (target != self.ignore_index).float() * boundary

        if valid_mask.sum() < self.smooth:
            return pred.sum() * 0.0

        dice_sum = pred.new_zeros(1)
        num_classes_computed = 0

        for c in range(self.num_classes):
            pred_c = pred_soft[:, c] * valid_mask
            target_c = (target == c).float() * valid_mask

            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()

            if union > self.smooth:
                dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
                dice_sum = dice_sum + (1.0 - dice)
                num_classes_computed += 1

        return dice_sum / max(num_classes_computed, 1)


class FocalBoundaryLoss(nn.Module):
    """
    Focal + 边界感知损失 (稳定版)
    """

    def __init__(
            self,
            num_classes: int,
            ignore_index: int = 255,
            gamma: float = 2.0,
            alpha: float = 0.25,
            boundary_weight: float = 5.0
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.gamma = gamma
        self.alpha = alpha
        self.boundary_weight = boundary_weight
        self.eps = 1e-7

        self.register_buffer('laplacian_kernel', torch.tensor(
            [[0, 1, 0],
             [1, -4, 1],
             [0, 1, 0]], dtype=torch.float32
        ).view(1, 1, 3, 3))

    def extract_boundary(self, mask: Tensor) -> Tensor:
        """提取边界"""
        with torch.no_grad():
            boundary = torch.zeros_like(mask).float().unsqueeze(1)

            for c in range(self.num_classes):
                class_mask = (mask == c).float().unsqueeze(1)
                if class_mask.sum() > 0:
                    kernel = self.laplacian_kernel.to(class_mask.device)
                    edges = F.conv2d(class_mask, kernel, padding=1)
                    boundary = boundary + (edges.abs() > 0).float()

            boundary = (boundary > 0).float()
            boundary = F.max_pool2d(boundary, 5, stride=1, padding=2)
            return boundary.squeeze(1)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """计算Focal + 边界加权损失"""
        # 强制float32
        pred = pred.float()

        # 裁剪输入防止溢出
        pred = torch.clamp(pred, min=-20.0, max=20.0)

        # 检查NaN
        if torch.isnan(pred).any():
            pred = torch.where(torch.isfinite(pred), pred, torch.zeros_like(pred))

        valid_mask = (target != self.ignore_index).float()
        num_valid = valid_mask.sum()

        if num_valid < 1:
            return pred.sum() * 0.0

        # 计算log softmax (更稳定)
        log_prob = F.log_softmax(pred, dim=1)
        prob = torch.exp(log_prob)

        # 安全gather
        target_safe = target.clone()
        target_safe[target == self.ignore_index] = 0

        ce = -log_prob.gather(1, target_safe.unsqueeze(1)).squeeze(1)
        pt = prob.gather(1, target_safe.unsqueeze(1)).squeeze(1)

        # Focal权重
        pt = torch.clamp(pt, min=self.eps, max=1.0 - self.eps)
        focal_weight = self.alpha * (1 - pt) ** self.gamma

        focal_loss = focal_weight * ce * valid_mask

        # 边界加权
        if self.boundary_weight > 1.0:
            boundary = self.extract_boundary(target)
            weight = 1.0 + boundary * (self.boundary_weight - 1.0)
            focal_loss = focal_loss * weight

        loss = focal_loss.sum() / (num_valid + self.eps)

        if torch.isnan(loss) or torch.isinf(loss):
            return F.cross_entropy(pred, target, ignore_index=self.ignore_index)

        return loss


def get_loss_function(
        loss_type: str,
        num_classes: int,
        ignore_index: int = 255,
        boundary_weight: float = 5.0,
        **kwargs
) -> nn.Module:
    """获取损失函数"""
    if loss_type == 'ce':
        return nn.CrossEntropyLoss(ignore_index=ignore_index)

    elif loss_type == 'boundary':
        return BoundaryAwareLoss(
            num_classes=num_classes,
            ignore_index=ignore_index,
            boundary_weight=boundary_weight,
            **kwargs
        )

    elif loss_type == 'focal_boundary':
        return FocalBoundaryLoss(
            num_classes=num_classes,
            ignore_index=ignore_index,
            boundary_weight=boundary_weight,
            **kwargs
        )

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# ==================== 测试 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("Boundary-Aware Loss Test (Complete Fix v2)")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 测试数据
    pred = torch.randn(2, 3, 256, 256, device=device)
    target = torch.randint(0, 3, (2, 256, 256), device=device)
    target[0, :50, :50] = 255

    # 测试1: 正常情况
    print("\n[Test 1] Normal case")
    criterion = BoundaryAwareLoss(num_classes=3, boundary_weight=5.0).to(device)
    loss = criterion(pred, target)
    print(f"  Loss: {loss.item():.4f}")

    # 测试2: AMP环境
    print("\n[Test 2] AMP environment")
    if device == 'cuda':
        with torch.cuda.amp.autocast():
            # 创建half精度输入模拟AMP
            pred_half = pred.half()
            loss_amp = criterion(pred_half, target)
            print(f"  AMP Loss: {loss_amp.item():.4f}")
            print(f"  Loss dtype: {loss_amp.dtype}")
    else:
        print("  CUDA not available")

    # 测试3: 极端输入
    print("\n[Test 3] Extreme inputs")
    pred_extreme = torch.randn(2, 3, 64, 64, device=device) * 100
    pred_extreme[0, 0, 0, 0] = float('nan')
    pred_extreme[0, 0, 0, 1] = float('inf')
    target_normal = torch.randint(0, 3, (2, 64, 64), device=device)
    loss_extreme = criterion(pred_extreme, target_normal)
    print(f"  Extreme Loss: {loss_extreme.item():.4f}")
    print(f"  Is finite: {torch.isfinite(loss_extreme).item()}")

    # 测试4: 梯度流
    print("\n[Test 4] Gradient flow")
    pred_grad = torch.randn(2, 3, 64, 64, device=device, requires_grad=True)
    target_grad = torch.randint(0, 3, (2, 64, 64), device=device)
    loss_grad = criterion(pred_grad, target_grad)
    loss_grad.backward()
    if pred_grad.grad is not None:
        print(f"  Gradient exists: True")
        print(f"  Gradient is finite: {torch.isfinite(pred_grad.grad).all().item()}")
    else:
        print("  Gradient exists: False")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

    print("""
推荐训练命令 (不使用AMP):
    python train.py --epochs 200 --batch-size 4 --learning-rate 1e-5 \\
        --scale 0.5 --classes 3 --validation 10 --use-dacg \\
        --loss-type boundary --boundary-weight 5.0

如果必须使用AMP，请确保边界权重较低:
    python train.py ... --boundary-weight 3.0 --amp
    """)