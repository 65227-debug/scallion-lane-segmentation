"""
推理脚本 - 兼容旧版本模型权重
支持 DSEM + DACG + TGSM 架构
自动处理权重不匹配和额外统计信息
"""

import argparse
import logging
import os
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from unet import UNet


def clean_state_dict(state_dict):
    """
    清理state_dict，移除额外的统计信息和不匹配的键
    """
    # 移除所有 total_ops 和 total_params
    keys_to_remove = [k for k in state_dict.keys() if 'total_ops' in k or 'total_params' in k]

    # 移除其他不需要的键
    unwanted_keys = ['mask_values', 'bridge.']
    for key in list(state_dict.keys()):
        if any(unwanted in key for unwanted in unwanted_keys):
            keys_to_remove.append(key)

    for key in keys_to_remove:
        if key in state_dict:
            del state_dict[key]

    return state_dict


def load_model_flexible(model, checkpoint_path, device):
    """
    灵活加载模型权重，自动处理不匹配
    支持从旧版本（无TGSM）到新版本（有TGSM）的迁移
    支持tgb -> tgsm键名映射
    """
    logging.info(f'Loading checkpoint from {checkpoint_path}')
    state_dict = torch.load(checkpoint_path, map_location=device)

    # 清理state_dict
    state_dict = clean_state_dict(state_dict)

    # 键名映射：tgb -> tgsm (兼容旧模型)
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

    # 获取模型的state_dict
    model_state = model.state_dict()

    # 尝试加载匹配的权重
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

    # 检查模型中缺失的键（新增的TGSM模块）
    for key in model_state.keys():
        if key not in state_dict:
            missing_keys.append(key)

    # 加载匹配的权重
    model.load_state_dict(new_state_dict, strict=False)

    logging.info(f'Successfully loaded {len(matched_keys)}/{len(state_dict)} parameters from checkpoint')

    if len(missing_keys) > 0:
        # 统计TGSM相关的缺失键
        tgsm_keys = [k for k in missing_keys if 'tgsm' in k.lower()]
        other_keys = [k for k in missing_keys if 'tgsm' not in k.lower()]

        if len(tgsm_keys) > 0:
            logging.warning(f'TGSM module not found in checkpoint ({len(tgsm_keys)} parameters will be randomly initialized)')
            logging.info('This is expected when loading old models without TGSM')

        if len(other_keys) > 0:
            logging.warning(f'{len(other_keys)} other parameters missing in checkpoint')
            # 只显示前5个
            for i, key in enumerate(other_keys[:5]):
                logging.warning(f'  - {key}')
            if len(other_keys) > 5:
                logging.warning(f'  ... and {len(other_keys) - 5} more')

    if len(unmatched_keys) > 0:
        logging.warning(f'Skipped {len(unmatched_keys)} unmatched parameters')
        # 只显示前5个不匹配的键
        for i, key in enumerate(unmatched_keys[:5]):
            logging.warning(f'  - {key}')
        if len(unmatched_keys) > 5:
            logging.warning(f'  ... and {len(unmatched_keys) - 5} more')

    return model


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

    # 标签映射：85->1, 170->2, 0->0
    if label_mapping is not None:
        mapped_mask = np.ones_like(mask) * 255
        for old_label, new_label in label_mapping.items():
            mapped_mask[mask == old_label] = new_label
        mask = mapped_mask

    return mask


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

    # mIoU
    valid_ious = [iou for iou in iou_list if iou > 0]
    ious['mIoU'] = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0

    # Lane mIoU (不含背景)
    lane_ious = iou_list[1:]
    valid_lane_ious = [iou for iou in lane_ious if iou > 0]
    ious['mIoU_Lane'] = sum(valid_lane_ious) / len(valid_lane_ious) if valid_lane_ious else 0.0

    return ious


def predict_image(model, image_tensor, device):
    """对单张图像进行推理"""
    model.eval()
    image_tensor = image_tensor.unsqueeze(0).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = model(image_tensor)
        pred = output.argmax(dim=1)[0].cpu().numpy()

    return pred


def save_prediction(pred, save_path, original_size=None):
    """保存预测结果，恢复原始标签值"""
    # 映射回原始标签值：0->0, 1->85, 2->170
    label_map = {0: 0, 1: 85, 2: 170}

    pred_original = np.zeros_like(pred, dtype=np.uint8)
    for new_label, old_label in label_map.items():
        pred_original[pred == new_label] = old_label

    # 转换为PIL图像
    pred_img = Image.fromarray(pred_original)

    # 如果需要，恢复到原始尺寸
    if original_size is not None:
        pred_img = pred_img.resize(original_size, resample=Image.NEAREST)

    pred_img.save(save_path)


def visualize_comparison(image_path, mask_path, pred, save_path, iou_scores):
    """可视化对比图：原图、真实标签、预测结果"""
    # 加载原图
    img = np.array(Image.open(image_path))

    # 加载真实标签
    mask = np.array(Image.open(mask_path))

    # 创建图形
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # 颜色映射
    colors = {
        0: [0, 0, 0],  # 背景 - 黑色
        85: [255, 0, 0],  # 左车道 - 红色
        170: [0, 255, 0],  # 右车道 - 绿色
    }

    # 创建彩色标签图
    def label_to_color(label_img):
        h, w = label_img.shape
        color_img = np.zeros((h, w, 3), dtype=np.uint8)
        for label_val, color in colors.items():
            color_img[label_img == label_val] = color
        return color_img

    # 预测结果映射回原始标签值
    label_map = {0: 0, 1: 85, 2: 170}
    pred_original = np.zeros_like(pred, dtype=np.uint8)
    for new_label, old_label in label_map.items():
        pred_original[pred == new_label] = old_label

    # 1. 原图
    axes[0].imshow(img)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # 2. 真实标签
    mask_color = label_to_color(mask)
    axes[1].imshow(mask_color)
    axes[1].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    # 3. 预测结果
    pred_color = label_to_color(pred_original)
    axes[2].imshow(pred_color)
    title = f"Prediction\nmIoU: {iou_scores['mIoU'] * 100:.2f}%, mIoU_Lane: {iou_scores['mIoU_Lane'] * 100:.2f}%"
    axes[2].set_title(title, fontsize=14, fontweight='bold')
    axes[2].axis('off')

    # 4. 叠加对比
    axes[3].imshow(img)
    axes[3].imshow(pred_color, alpha=0.5)
    axes[3].set_title('Overlay', fontsize=14, fontweight='bold')
    axes[3].axis('off')

    # 添加图例
    legend_elements = [
        mpatches.Patch(facecolor='black', label='Background'),
        mpatches.Patch(facecolor='red', label='Left Lane'),
        mpatches.Patch(facecolor='green', label='Right Lane')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=12)

    # 添加详细IoU信息
    iou_text = f"Background: {iou_scores['IoU_Background'] * 100:.2f}%  |  "
    iou_text += f"Left Lane: {iou_scores['IoU_Left_Lane'] * 100:.2f}%  |  "
    iou_text += f"Right Lane: {iou_scores['IoU_Right_Lane'] * 100:.2f}%"
    fig.text(0.5, 0.02, iou_text, ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='推理并评估模型（兼容旧权重，支持TGSM）')
    parser.add_argument('--image-dir', type=str,
                        default='data/images',
                        help='原图路径')
    parser.add_argument('--mask-dir', type=str,
                        default='data/masks',
                        help='标签路径')
    parser.add_argument('--model-path', type=str,
                        required=True,
                        help='模型权重路径')
    parser.add_argument('--output-dir', type=str,
                        default='./inference_results',
                        help='输出路径')
    parser.add_argument('--scale', type=float, default=0.5,
                        help='图像缩放比例')
    parser.add_argument('--num-worst', type=int, default=20,
                        help='保存最差的N张图像')
    parser.add_argument('--classes', type=int, default=3,
                        help='类别数')
    parser.add_argument('--use-dsem', action='store_true', default=False,
                        help='模型是否使用DSEM')
    parser.add_argument('--use-dacg', action='store_true', default=False,
                        help='模型是否使用DACG')
    parser.add_argument('--use-tgsm', action='store_true', default=False,
                        help='模型是否使用TGSM (Topology-Guided Spatial Module)')
    parser.add_argument('--tgsm-version', type=str, default='enhanced',
                        choices=['basic', 'enhanced', 'advanced'],
                        help='TGSM版本 (仅当use-tgsm=True时有效)')
    parser.add_argument('--bilinear', action='store_true', default=False,
                        help='模型是否使用双线性上采样')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # 创建输出目录
    output_dir = Path(args.output_dir)
    pred_dir = output_dir / 'predictions'
    worst_dir = output_dir / 'worst_cases'
    pred_dir.mkdir(parents=True, exist_ok=True)
    worst_dir.mkdir(parents=True, exist_ok=True)

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

    # 使用灵活加载
    model = load_model_flexible(model, args.model_path, device)
    model.to(device=device)

    logging.info('Model loaded successfully')

    # 获取所有图像文件
    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)

    image_files = sorted(list(image_dir.glob('*.jpg')) +
                         list(image_dir.glob('*.png')) +
                         list(image_dir.glob('*.jpeg')))

    if len(image_files) == 0:
        logging.error(f'No images found in {image_dir}')
        return

    logging.info(f'Found {len(image_files)} images')

    # 标签映射
    label_mapping = {0: 0, 85: 1, 170: 2}

    # 存储每张图像的IoU分数
    results = []

    # 推理所有图像
    logging.info('Starting inference...')
    for img_path in tqdm(image_files, desc='Processing images'):
        # 获取对应的标签文件
        img_name = img_path.stem
        mask_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            potential_mask = mask_dir / f"{img_name}{ext}"
            if potential_mask.exists():
                mask_path = potential_mask
                break

        if mask_path is None:
            logging.warning(f'Mask not found for {img_name}, skipping...')
            continue

        try:
            # 加载图像和标签
            image = load_image(img_path, scale=args.scale)
            mask_gt = load_mask(mask_path, scale=args.scale, label_mapping=label_mapping)

            # 预测
            pred = predict_image(model, image, device)

            # 计算IoU
            iou_scores = calculate_iou_per_image(pred, mask_gt, num_classes=args.classes)

            # 保存预测结果（恢复到原始尺寸和原始标签值）
            original_img = Image.open(img_path)
            save_prediction(pred, pred_dir / f"{img_name}.png", original_size=original_img.size)

            # 记录结果
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

    # 计算整体统计
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

        # 找出效果最差的图像
        results_sorted = sorted(results, key=lambda x: x['mIoU_Lane'])
        worst_cases = results_sorted[:args.num_worst]

        logging.info('\n' + '=' * 60)
        logging.info(f'Top {args.num_worst} Worst Cases (by mIoU_Lane):')
        logging.info('=' * 60)

        # 保存最差案例的详细信息
        worst_info_file = worst_dir / 'worst_cases_info.txt'
        with open(worst_info_file, 'w') as f:
            f.write('=' * 80 + '\n')
            f.write(f'Model Configuration: {config_str}\n')
            f.write(f'Top {args.num_worst} Worst Cases (Sorted by mIoU_Lane)\n')
            f.write('=' * 80 + '\n\n')

            for i, case in enumerate(worst_cases, 1):
                info = f"[{i}] {case['image_name']}\n"
                info += f"    mIoU: {case['mIoU'] * 100:.2f}%\n"
                info += f"    mIoU_Lane: {case['mIoU_Lane'] * 100:.2f}%\n"
                info += f"    IoU_Background: {case['iou_scores']['IoU_Background'] * 100:.2f}%\n"
                info += f"    IoU_Left_Lane: {case['iou_scores']['IoU_Left_Lane'] * 100:.2f}%\n"
                info += f"    IoU_Right_Lane: {case['iou_scores']['IoU_Right_Lane'] * 100:.2f}%\n"
                info += '-' * 80 + '\n'

                logging.info(info)
                f.write(info + '\n')

                # 生成可视化对比图
                vis_path = worst_dir / f'worst_{i:03d}_{case["image_name"]}.png'
                visualize_comparison(
                    case['image_path'],
                    case['mask_path'],
                    case['pred'],
                    vis_path,
                    case['iou_scores']
                )

        logging.info(f'\nWorst cases visualization saved to {worst_dir}')
        logging.info(f'Detailed info saved to {worst_info_file}')

        # 保存所有结果的CSV
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

        # 统计分布
        logging.info('\n' + '=' * 60)
        logging.info('IoU Distribution:')
        logging.info('=' * 60)

        miou_lane_values = [r['mIoU_Lane'] for r in results]
        ranges = [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]

        for low, high in ranges:
            count = sum(1 for v in miou_lane_values if low <= v < high)
            percentage = count / len(miou_lane_values) * 100
            logging.info(f'  {low * 100:.0f}%-{high * 100:.0f}%: {count} images ({percentage:.1f}%)')

    logging.info('\n' + '=' * 60)
    logging.info('Inference and evaluation completed!')
    logging.info('=' * 60)
    logging.info(f'\nOutput directory: {output_dir}')
    logging.info(f'  - Predictions: {pred_dir}')
    logging.info(f'  - Worst cases: {worst_dir}')
    logging.info(f'  - CSV report: {output_dir}/all_results.csv')


if __name__ == '__main__':
    main()
