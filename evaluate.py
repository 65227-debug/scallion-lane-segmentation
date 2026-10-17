import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff


def map_mask_labels(mask, label_mapping):
    """
    将掩码中的原始标签值映射到连续的类别索引

    Args:
        mask: 原始掩码 tensor [B, H, W] 或 [H, W]
        label_mapping: 字典 {原始值: 目标索引}，例如 {0: 0, 85: 1, 170: 2}

    Returns:
        映射后的掩码
    """
    mapped_mask = torch.zeros_like(mask)
    for original_value, target_index in label_mapping.items():
        mapped_mask[mask == original_value] = target_index
    return mapped_mask


@torch.inference_mode()
def evaluate(net, dataloader, device, amp, label_mapping=None):
    """
    评估网络性能

    Args:
        net: 神经网络模型
        dataloader: 数据加载器
        device: 设备 (cuda/cpu)
        amp: 是否使用混合精度
        label_mapping: 标签映射字典，例如 {0: 0, 85: 1, 170: 2}
                      如果为 None，则假设标签已经是 [0, 1, 2, ...]
    """
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # 如果没有提供标签映射，创建一个恒等映射
    if label_mapping is None:
        # 假设标签已经是 [0, 1, 2, ...]
        label_mapping = {i: i for i in range(net.n_classes)}

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # 映射标签值到类别索引
            mask_true = map_mask_labels(mask_true, label_mapping)

            # predict the mask
            mask_pred = net(image)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                # 检查映射后的标签是否在有效范围内
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, \
                    f'True mask indices should be in [0, {net.n_classes}), but got [{mask_true.min()}, {mask_true.max()}]'

                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    net.train()
    return dice_score / max(num_val_batches, 1)