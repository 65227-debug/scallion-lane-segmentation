"""
快速评估脚本 - 修复版 (支持自动清理checkpoint)
直接计算Kappa、FPS、mIoU、MPA、Params

使用方法:
    python eval_fixed.py --checkpoint model.pth --classes 3 --use-dsem --use-dacg
"""

import torch
import numpy as np
import time
import argparse
from pathlib import Path
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image
import sys

# 导入项目模块
from unet import UNet


def clean_state_dict(state_dict):
    """清理state_dict，移除统计参数"""
    # 移除mask_values
    if 'mask_values' in state_dict:
        del state_dict['mask_values']

    # 移除total_ops和total_params
    keys_to_remove = [k for k in list(state_dict.keys())
                     if 'total_ops' in k or 'total_params' in k]

    for key in keys_to_remove:
        del state_dict[key]

    return state_dict


def load_model_smart(checkpoint_path, device, args):
    """智能加载模型，自动检测配置"""
    print(f"\n📦 Loading checkpoint: {checkpoint_path}")

    # 加载checkpoint
    state_dict = torch.load(checkpoint_path, map_location=device)

    # 清理state_dict
    state_dict = clean_state_dict(state_dict)

    # 检测模型配置
    has_tgsm = any('tgsm' in k for k in state_dict.keys())
    has_dsem = any('dir_convs' in k for k in state_dict.keys())
    has_dacg = any('dacg' in k for k in state_dict.keys())

    print(f"   Detected: DSEM={'✅' if has_dsem else '❌'}, "
          f"DACG={'✅' if has_dacg else '❌'}, "
          f"TGSM={'✅' if has_tgsm else '❌'}")

    # 自动调整配置
    use_tgsm = args.use_tgsm or has_tgsm
    use_dsem = args.use_dsem or has_dsem
    use_dacg = args.use_dacg or has_dacg

    if has_tgsm and not args.use_tgsm:
        print(f"   ⚠️  Auto-enabling TGSM (detected in checkpoint)")

    # 创建模型
    model = UNet(
        n_channels=3,
        n_classes=args.classes,
        bilinear=args.bilinear,
        use_dsem=use_dsem,
        reduction=args.reduction,
        use_dacg=use_dacg,
        use_tgsm=use_tgsm,
        tgsm_version=args.tgsm_version
    )

    # 加载权重
    try:
        model.load_state_dict(state_dict, strict=True)
        print(f"   ✅ Loaded successfully (strict mode)")
    except RuntimeError as e:
        print(f"   ⚠️  Trying non-strict mode...")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"   ✅ Loaded (non-strict: {len(missing)} missing, {len(unexpected)} unexpected)")

    model.to(device)
    model.eval()

    return model


def count_parameters(model):
    """计算参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total / 1e6, trainable / 1e6


def compute_kappa(model, dataloader, device, num_classes=3):
    """计算Kappa系数"""
    print("\n🔍 Computing Cohen's Kappa...")
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Kappa'):
            images = batch['image'].to(device, dtype=torch.float32)
            masks = batch['mask'].to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            pred_flat = preds.cpu().numpy().flatten()
            true_flat = masks.cpu().numpy().flatten()

            valid = (true_flat != 255)
            all_preds.extend(pred_flat[valid])
            all_targets.extend(true_flat[valid])

    kappa = cohen_kappa_score(all_targets, all_preds)
    return kappa


def compute_fps(model, device, input_size=(1, 3, 256, 256), iterations=100):
    """计算FPS"""
    print("\n⚡ Measuring FPS...")
    model.eval()
    dummy = torch.randn(*input_size).to(device)

    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    # 测量
    times = []
    with torch.no_grad():
        for _ in tqdm(range(iterations), desc='FPS'):
            start = time.time()
            _ = model(dummy)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.time() - start)

    avg_time = np.mean(times)
    return 1.0 / avg_time, avg_time * 1000


def compute_metrics(model, dataloader, device, num_classes=3):
    """计算IoU和MPA"""
    print("\n📊 Computing IoU and MPA...")
    model.eval()

    intersection = torch.zeros(num_classes, device=device)
    union = torch.zeros(num_classes, device=device)
    total_target = torch.zeros(num_classes, device=device)
    total_correct = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Metrics'):
            images = batch['image'].to(device, dtype=torch.float32)
            masks = batch['mask'].to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            valid = (masks != 255)

            for c in range(num_classes):
                pred_c = (preds == c) & valid
                true_c = (masks == c) & valid

                inter = (pred_c & true_c).sum().float()
                union_c = (pred_c | true_c).sum().float()

                intersection[c] += inter
                union[c] += union_c
                total_target[c] += true_c.sum().float()
                total_correct[c] += inter

    # 计算IoU
    ious = []
    for c in range(num_classes):
        if union[c] > 0:
            ious.append((intersection[c] / union[c]).item() * 100)
        else:
            ious.append(0.0)

    # 计算MPA
    pas = []
    for c in range(num_classes):
        if total_target[c] > 0:
            pas.append((total_correct[c] / total_target[c]).item() * 100)

    mpa = np.mean(pas) if pas else 0.0
    miou = np.mean(ious)
    miou_lane = np.mean(ious[1:]) if len(ious) > 1 else 0.0

    return ious, miou, miou_lane, mpa


def print_results(params, kappa, fps, latency, ious, miou, miou_lane, mpa, class_names):
    """打印结果"""
    print("\n" + "="*60)
    print("📋 EVALUATION RESULTS".center(60))
    print("="*60)

    print(f"\n🔧 Model Parameters:")
    print(f"   Total:          {params[0]:.2f} M")
    print(f"   Trainable:      {params[1]:.2f} M")

    print(f"\n✨ Cohen's Kappa:  {kappa:.4f}")
    if kappa >= 0.80:
        print(f"   → Almost perfect agreement! 🎉")
    elif kappa >= 0.60:
        print(f"   → Substantial agreement ✓")
    else:
        print(f"   → Moderate/Fair agreement")

    print(f"\n⚡ Inference Speed:")
    print(f"   FPS:            {fps:.2f} frames/sec")
    print(f"   Latency:        {latency:.2f} ms")

    print(f"\n📊 IoU Metrics:")
    for name, iou in zip(class_names, ious):
        print(f"   {name:15s}: {iou:6.2f}%")
    print(f"   {'mIoU (all)':15s}: {miou:6.2f}%")
    print(f"   {'mIoU (lane)':15s}: {miou_lane:6.2f}%")

    print(f"\n🎯 Pixel Accuracy:")
    print(f"   MPA:            {mpa:.2f}%")

    print("\n" + "="*60)
    print("📊 Summary Table")
    print("="*60)
    print(f"{'Metric':<20} {'Value':<15}")
    print("-"*35)
    print(f"{'Params (M)':<20} {params[0]:<15.2f}")
    print(f"{'mIoU (%)':<20} {miou:<15.2f}")
    print(f"{'mIoU_Lane (%)':<20} {miou_lane:<15.2f}")
    print(f"{'MPA (%)':<20} {mpa:<15.2f}")
    print(f"{'Kappa':<20} {kappa:<15.4f}")
    print(f"{'FPS':<20} {fps:<15.2f}")

    print("\n📝 LaTeX Row (完整):")
    print(f"DAG-UNet & {params[0]:.2f} & " + " & ".join(f"{iou:.2f}" for iou in ious) +
          f" & {miou:.2f} & {miou_lane:.2f} & {mpa:.2f} & {kappa:.4f} & {fps:.2f} \\\\")

    print("\n📝 LaTeX Row (简化):")
    print(f"DAG-UNet & {params[0]:.2f} & {miou:.2f} & {mpa:.2f} & {kappa:.4f} & {fps:.2f} \\\\")

    print("\n" + "="*60 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--classes', type=int, default=3)
    parser.add_argument('--bilinear', action='store_true')
    parser.add_argument('--reduction', type=int, default=4)
    parser.add_argument('--use-dsem', action='store_true')
    parser.add_argument('--use-dacg', action='store_true')
    parser.add_argument('--use-tgsm', action='store_true')
    parser.add_argument('--tgsm-version', type=str, default='enhanced')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--scale', type=float, default=0.5)
    parser.add_argument('--data-dir', type=str, default='data/images')
    parser.add_argument('--mask-dir', type=str, default='data/masks')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_names = ['Background', 'Left_Lane', 'Right_Lane'][:args.classes]

    print("="*60)
    print("DAG-UNet Evaluation (Fixed Version)".center(60))
    print("="*60)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    # 加载模型
    model = load_model_smart(args.checkpoint, device, args)

    # 计算参数量
    print("\n🔧 Computing parameters...")
    params = count_parameters(model)
    print(f"   Total: {params[0]:.2f}M, Trainable: {params[1]:.2f}M")

    # 加载数据
    print("\n📂 Loading dataset...")
    data_dir = Path(args.data_dir)
    mask_dir = Path(args.mask_dir)

    # 简化的数据集加载 - 避免文件名匹配问题
    from PIL import Image
    from torch.utils.data import Dataset

    class SimpleDataset(Dataset):
        def __init__(self, img_dir, mask_dir, scale=0.5):
            self.img_dir = Path(img_dir)
            self.mask_dir = Path(mask_dir)
            self.scale = scale

            # 获取所有图像文件
            self.images = sorted(list(self.img_dir.glob('*.png')) +
                               list(self.img_dir.glob('*.jpg')))

            print(f"   Found {len(self.images)} images in {img_dir}")

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            img_path = self.images[idx]

            # 尝试不同的mask文件名模式
            img_name = img_path.stem
            possible_masks = [
                self.mask_dir / f"{img_name}.png",
                self.mask_dir / f"{img_name}.jpg",
                self.mask_dir / f"{img_name}_mask.png",
                self.mask_dir / f"{img_name}_label.png",
            ]

            mask_path = None
            for p in possible_masks:
                if p.exists():
                    mask_path = p
                    break

            if mask_path is None:
                # 如果找不到对应的mask，跳过
                return self.__getitem__((idx + 1) % len(self))

            # 加载图像和mask
            img = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')

            # 调整大小
            w, h = img.size
            newW, newH = int(self.scale * w), int(self.scale * h)

            img = img.resize((newW, newH), Image.BILINEAR)
            mask = mask.resize((newW, newH), Image.NEAREST)

            # 转换为tensor
            img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(np.array(mask)).long()

            # 映射标签值 [0, 85, 170] -> [0, 1, 2]
            mask_mapped = torch.full_like(mask, 255)
            mask_mapped[mask == 0] = 0
            mask_mapped[mask == 85] = 1
            mask_mapped[mask == 170] = 2

            return {'image': img, 'mask': mask_mapped}

    try:
        dataset = SimpleDataset(data_dir, mask_dir, args.scale)
    except Exception as e:
        print(f"   ❌ Error loading dataset: {e}")
        print(f"   Please check:")
        print(f"      - Images directory: {data_dir}")
        print(f"      - Masks directory: {mask_dir}")
        print(f"      - File naming convention")
        sys.exit(1)

    # 使用子集加快评估
    subset_size = min(200, len(dataset))
    indices = list(range(0, len(dataset), max(1, len(dataset) // subset_size)))[:subset_size]
    subset = torch.utils.data.Subset(dataset, indices)

    dataloader = DataLoader(subset, batch_size=args.batch_size,
                          shuffle=False, num_workers=0)
    print(f"   Using {len(subset)} images")

    # 评估
    print("\n" + "="*60)
    print("Running Evaluation...".center(60))
    print("="*60)

    kappa = compute_kappa(model, dataloader, device, args.classes)
    fps, latency = compute_fps(model, device)
    ious, miou, miou_lane, mpa = compute_metrics(model, dataloader, device, args.classes)

    # 打印结果
    print_results(params, kappa, fps, latency, ious, miou, miou_lane, mpa, class_names)


if __name__ == '__main__':
    main()
