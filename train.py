"""
DAG-Net Training Script with Per-Class IoU Evaluation (DSEM版本)
集成各类别IoU评估的训练脚本 - 使用DSEM替代SCSCA

评估指标：
- mIoU: 平均IoU（所有类别）
- mIoU_Lane: 平均IoU（不含背景，仅车道线）
- IoU_Background: 背景IoU
- IoU_Left_Lane: 左车道线IoU
- IoU_Right_Lane: 右车道线IoU
- MPA: 平均像素准确率
- Dice: Dice系数

支持模块：
- DSEM: 方向感知空间增强模块（编码器）
- DACG: 方向感知上下文门控（解码器）
- TGSM: 拓扑引导空间模块（bottleneck）
"""

import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from utils.boundary_loss import BoundaryAwareLoss, FocalBoundaryLoss, get_loss_function
from utils.metrics import (
    SegmentationMetrics,
    calculate_class_iou,
    calculate_miou,
    calculate_mpa,
    count_parameters,
    calculate_flops,
    calculate_fps
)

# 类别名称定义
CLASS_NAMES = ['Background', 'Left_Lane', 'Right_Lane']


def evaluate_with_metrics(model, dataloader, device, amp, num_classes):
    """
    评估模型并计算所有指标（包括各类别IoU）

    Returns:
        dict: 包含所有评估指标的字典
    """
    model.eval()
    num_val_batches = len(dataloader)

    dice_score = 0
    class_names = CLASS_NAMES[:num_classes]

    total_intersection = torch.zeros(num_classes, device=device)
    total_union = torch.zeros(num_classes, device=device)
    total_target = torch.zeros(num_classes, device=device)
    total_correct = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
            for batch in tqdm(dataloader, total=num_val_batches, desc='Validation', unit='batch', leave=False):
                image, mask_true = batch['image'], batch['mask']

                image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                mask_true = mask_true.to(device=device, dtype=torch.long)

                mask_pred = model(image)
                pred_classes = mask_pred.argmax(dim=1)

                valid_mask = (mask_true != 255)

                if num_classes == 1:
                    mask_pred_binary = (F.sigmoid(mask_pred) > 0.5).float()
                    from utils.dice_score import dice_coeff
                    dice_score += dice_coeff(mask_pred_binary, mask_true.float(), reduce_batch_first=False)
                else:
                    from utils.dice_score import multiclass_dice_coeff

                    mask_true_clamped = mask_true.clone()
                    mask_true_clamped[~valid_mask] = 0

                    mask_true_onehot = F.one_hot(
                        mask_true_clamped, num_classes
                    ).permute(0, 3, 1, 2).float()
                    mask_pred_onehot = F.one_hot(
                        pred_classes, num_classes
                    ).permute(0, 3, 1, 2).float()

                    dice_score += multiclass_dice_coeff(
                        mask_pred_onehot[:, 1:], mask_true_onehot[:, 1:],
                        reduce_batch_first=False
                    )

                for c in range(num_classes):
                    pred_c = (pred_classes == c) & valid_mask
                    target_c = (mask_true == c) & valid_mask

                    intersection = (pred_c & target_c).sum().float()
                    union = (pred_c | target_c).sum().float()

                    total_intersection[c] += intersection
                    total_union[c] += union
                    total_target[c] += target_c.sum().float()
                    total_correct[c] += intersection

    model.train()

    results = {}

    avg_dice = dice_score / max(num_val_batches, 1)
    results['Dice'] = avg_dice.item() if torch.is_tensor(avg_dice) else avg_dice

    iou_list = []
    iou_lane_list = []

    for c in range(num_classes):
        if total_union[c] > 0:
            iou = (total_intersection[c] / total_union[c]).item() * 100
        else:
            iou = 0.0

        class_name = class_names[c] if c < len(class_names) else f'Class_{c}'
        results[f'IoU_{class_name}'] = iou
        iou_list.append(iou)

        if c > 0:
            iou_lane_list.append(iou)

    valid_ious = [iou for iou in iou_list if iou > 0]
    results['mIoU'] = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0

    valid_lane_ious = [iou for iou in iou_lane_list if iou > 0]
    results['mIoU_Lane'] = sum(valid_lane_ious) / len(valid_lane_ious) if valid_lane_ious else 0.0

    pa_list = []
    for c in range(num_classes):
        if total_target[c] > 0:
            pa = (total_correct[c] / total_target[c]).item() * 100
            pa_list.append(pa)

    results['MPA'] = sum(pa_list) / len(pa_list) if pa_list else 0.0

    return results


def train_model(
        model,
        device,
        image_dir: Path,
        mask_dir: Path,
        checkpoint_dir: Path,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        loss_type: str = 'boundary',
        boundary_weight: float = 20.0,
        use_boundary_loss: bool = True,
        boundary_loss_weight: float = 0.5,
):
    # 定义原始标签值（用于数据加载）
    original_labels = [0, 85, 170]

    logging.info(f'Original labels for data loading: {original_labels}')
    logging.info(f'Class names: {CLASS_NAMES}')

    # 1. Create dataset
    try:
        dataset = CarvanaDataset(image_dir, mask_dir, img_scale,
                                 selected_classes=original_labels)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(image_dir, mask_dir, img_scale,
                               selected_classes=original_labels)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # 计算模型静态指标
    logging.info('Calculating model metrics...')
    total_params = count_parameters(model)
    flops = calculate_flops(model, input_size=(1, 3, 256, 256), device=device)
    try:
        fps = calculate_fps(model, input_size=(1, 3, 256, 256), device=device)
    except TypeError:
        # 如果函数不支持 num_iterations 参数，使用默认调用
        fps = calculate_fps(model, device=device)

    logging.info(f'Model Metrics:')
    logging.info(f'  Parameters: {total_params[0]:.2f} M')
    logging.info(f'  FLOPs: {flops:.2f} G')
    logging.info(f'  FPS: {fps:.2f} f/s')

    # 初始化WandB
    # 禁用 wandb
    os.environ['WANDB_MODE'] = 'disabled'
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must', mode='disabled')

    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp,
             loss_type=loss_type, boundary_weight=boundary_weight,
             use_boundary_loss=use_boundary_loss, boundary_loss_weight=boundary_loss_weight,
             total_params=total_params, flops=flops, fps=fps)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
        Loss type:       {loss_type}
        Boundary weight: {boundary_weight}
    ''')

    # 4. Set up optimizer, loss, scheduler
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # 损失函数
    if loss_type == 'ce':
        criterion = nn.CrossEntropyLoss(ignore_index=255) if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    elif loss_type == 'boundary':
        criterion = BoundaryAwareLoss(
            num_classes=model.n_classes,
            ignore_index=255,
            boundary_weight=boundary_weight,
            use_boundary_loss=use_boundary_loss,
            boundary_loss_weight=boundary_loss_weight
        )
    elif loss_type == 'focal_boundary':
        criterion = FocalBoundaryLoss(
            num_classes=model.n_classes,
            ignore_index=255,
            boundary_weight=boundary_weight
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        # loss += dice_loss(
                        #     F.softmax(masks_pred, dim=1).float(),
                        #     F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                        #     multiclass=True
                        # )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_results = evaluate_with_metrics(
                            model, val_loader, device, amp, model.n_classes
                        )

                        scheduler.step(val_results['mIoU_Lane'])

                        log_msg = (
                            f"Validation - "
                            f"Dice: {val_results['Dice']:.4f}, "
                            f"mIoU: {val_results['mIoU']:.2f}%, "
                            f"mIoU_Lane: {val_results['mIoU_Lane']:.2f}%, "
                        )

                        for c, name in enumerate(CLASS_NAMES[:model.n_classes]):
                            log_msg += f"IoU_{name}: {val_results.get(f'IoU_{name}', 0):.2f}%, "

                        log_msg += f"MPA: {val_results['MPA']:.2f}%"
                        logging.info(log_msg)

                        try:
                            wandb_log = {
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_results['Dice'],
                                'validation mIoU (%)': val_results['mIoU'],
                                'validation mIoU_Lane (%)': val_results['mIoU_Lane'],
                                'validation MPA (%)': val_results['MPA'],
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                            }

                            for c, name in enumerate(CLASS_NAMES[:model.n_classes]):
                                wandb_log[f'validation IoU_{name} (%)'] = val_results.get(f'IoU_{name}', 0)

                            wandb_log.update(histograms)
                            experiment.log(wandb_log)
                        except:
                            pass

        if save_checkpoint:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(checkpoint_dir / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')

    # 训练结束 - 最终评估
    logging.info("\n" + "=" * 60)
    logging.info("Running final evaluation...")
    final_results = evaluate_with_metrics(model, val_loader, device, amp, model.n_classes)

    class_iou_str = ""
    for c, name in enumerate(CLASS_NAMES[:model.n_classes]):
        class_iou_str += f"\n      IoU_{name}:        {final_results.get(f'IoU_{name}', 0):.2f}%"

    logging.info(f'''
    ============ Final Evaluation Results ============

    Dice Score:          {final_results['Dice']:.4f}

    IoU Metrics:
      mIoU (all):        {final_results['mIoU']:.2f}%
      mIoU (lane only):  {final_results['mIoU_Lane']:.2f}%{class_iou_str}

    Accuracy Metrics:
      MPA:               {final_results['MPA']:.2f}%

    Model Metrics:
      Parameters:        {total_params:.2f} M
      FLOPs:             {flops:.2f} G
      FPS:               {fps:.2f} f/s

    ================================================
    ''')

    logging.info("LaTeX Table Row (for paper):")
    iou_bg = final_results.get('IoU_Background', 0)
    iou_left = final_results.get('IoU_Left_Lane', 0)
    iou_right = final_results.get('IoU_Right_Lane', 0)
    logging.info(f"Method & {iou_bg:.2f} & {iou_left:.2f} & {iou_right:.2f} & "
                 f"{final_results['mIoU']:.2f} & {final_results['mIoU_Lane']:.2f} & "
                 f"{final_results['MPA']:.2f} \\\\")


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=3, help='Number of classes')
    parser.add_argument('--image-dir', type=Path, default=Path('data/images'),
                        help='Directory containing training images')
    parser.add_argument('--mask-dir', type=Path, default=Path('data/masks'),
                        help='Directory containing segmentation masks')
    parser.add_argument('--checkpoint-dir', type=Path, default=Path('checkpoints'),
                        help='Directory used to save checkpoints')

    # DSEM (替代 SCSCA)
    parser.add_argument('--use-dsem', action='store_true', default=False,
                        help='Use DSEM (Direction-aware Spatial Enhancement Module) in encoder')
    parser.add_argument('--reduction', '-r', type=int, default=4,
                        help='DSEM channel reduction ratio (default: 4)')

    # DACG
    parser.add_argument('--use-dacg', action='store_true', default=False,
                        help='Use DACG (Direction-Aware Context Gating) in decoder')

    # TGSM (拓扑引导空间模块)
    parser.add_argument('--use-tgsm', action='store_true', default=False,
                        help='Use TGSM (Topology-Guided Spatial Module) at bottleneck')
    parser.add_argument('--tgsm-version', type=str, default='enhanced',
                        choices=['basic', 'enhanced', 'advanced'],
                        help='TGSM version: basic(快速), enhanced(推荐), advanced(高级)')

    # Loss function
    parser.add_argument('--loss-type', type=str, default='boundary',
                        choices=['ce', 'boundary', 'focal_boundary'],
                        help='Loss function type')
    parser.add_argument('--boundary-weight', type=float, default=20.0,
                        help='Boundary region loss weight multiplier')
    parser.add_argument('--use-boundary-loss', action='store_true', default=True,
                        help='Use additional boundary loss')
    parser.add_argument('--no-boundary-loss', action='store_false', dest='use_boundary_loss',
                        help='Disable additional boundary loss')
    parser.add_argument('--boundary-loss-weight', type=float, default=0.5,
                        help='Weight for additional boundary loss')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # 创建模型（支持 DSEM + DACG + TGSM）
    model = UNet(
        n_channels=3,
        n_classes=args.classes,
        bilinear=args.bilinear,
        use_dsem=args.use_dsem,
        reduction=args.reduction,
        use_dacg=args.use_dacg,
        use_tgsm=args.use_tgsm,
        tgsm_version=args.tgsm_version
    )
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes): {CLASS_NAMES[:model.n_classes]}\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)

    try:
        train_model(
            model=model,
            image_dir=args.image_dir,
            mask_dir=args.mask_dir,
            checkpoint_dir=args.checkpoint_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            loss_type=args.loss_type,
            boundary_weight=args.boundary_weight,
            use_boundary_loss=args.use_boundary_loss,
            boundary_loss_weight=args.boundary_loss_weight,
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! Enabling checkpointing...')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            image_dir=args.image_dir,
            mask_dir=args.mask_dir,
            checkpoint_dir=args.checkpoint_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            loss_type=args.loss_type,
            boundary_weight=args.boundary_weight,
            use_boundary_loss=args.use_boundary_loss,
            boundary_loss_weight=args.boundary_loss_weight,
        )
