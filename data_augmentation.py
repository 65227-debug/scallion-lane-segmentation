import argparse
import cv2
import numpy as np
import os
from pathlib import Path


def adjust_brightness_contrast(image, brightness_factor=1.0, contrast_factor=1.0):
    """
    调整图像的亮度和对比度

    参数:
        image: 输入图像
        brightness_factor: 亮度因子 (0.8~1.2, 1.0为原始亮度)
        contrast_factor: 对比度因子 (0.8~1.2, 1.0为原始对比度)

    返回:
        调整后的图像
    """
    # 转换为float进行计算
    img_float = image.astype(np.float32)

    # 调整对比度：(pixel - 128) * contrast + 128
    img_float = (img_float - 128) * contrast_factor + 128

    # 调整亮度：pixel * brightness
    img_float = img_float * brightness_factor

    # 裁剪到0-255并转回uint8
    img_float = np.clip(img_float, 0, 255)

    return img_float.astype(np.uint8)


def augment_image(image, mask=None, seed=None):
    """
    对图像进行数据增强 - 包含几何变换和亮度/对比度调整

    参数:
        image: 输入图像 (numpy array) - 干净的原图
        mask: 对应的掩码 (numpy array)，如果为None则只处理图像
        seed: 随机种子，用于保证图像和掩码使用相同的随机变换

    返回:
        augmented_images: 包含所有增强版本的字典
        augmented_masks: 包含所有增强掩码的字典（如果提供了mask）
    """
    if seed is not None:
        np.random.seed(seed)

    augmented_images = {}
    augmented_masks = {} if mask is not None else None

    h, w = image.shape[:2]

    # ========== 基于干净原图的几何变换 ==========

    # 1. 原图（干净的）
    augmented_images['original'] = image.copy()
    if mask is not None:
        augmented_masks['original'] = mask.copy()

    # 2. 水平镜像（基于干净原图）
    augmented_images['mirrored'] = cv2.flip(image, 1)
    if mask is not None:
        augmented_masks['mirrored'] = cv2.flip(mask, 1)

    # 3. 随机旋转 (-5° ~ 5°)（基于干净原图）
    angle = np.random.uniform(-5, 5)
    M_rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    augmented_images['rotated'] = cv2.warpAffine(image, M_rot, (w, h),
                                                 flags=cv2.INTER_LINEAR,
                                                 borderMode=cv2.BORDER_REFLECT)
    if mask is not None:
        augmented_masks['rotated'] = cv2.warpAffine(mask, M_rot, (w, h),
                                                    flags=cv2.INTER_NEAREST,
                                                    borderMode=cv2.BORDER_REFLECT)

    # 4. 透视变换（轻度）（基于干净原图） - 车道线检测最关键
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    # 随机扰动四个角点
    offset = int(w * 0.05)  # 5% 的轻度扰动
    pts2 = np.float32([
        [np.random.randint(-offset, offset), np.random.randint(-offset, offset)],
        [w + np.random.randint(-offset, offset), np.random.randint(-offset, offset)],
        [np.random.randint(-offset, offset), h + np.random.randint(-offset, offset)],
        [w + np.random.randint(-offset, offset), h + np.random.randint(-offset, offset)]
    ])
    M_pers = cv2.getPerspectiveTransform(pts1, pts2)
    augmented_images['perspective'] = cv2.warpPerspective(image, M_pers, (w, h),
                                                          flags=cv2.INTER_LINEAR,
                                                          borderMode=cv2.BORDER_REFLECT)
    if mask is not None:
        augmented_masks['perspective'] = cv2.warpPerspective(mask, M_pers, (w, h),
                                                             flags=cv2.INTER_NEAREST,
                                                             borderMode=cv2.BORDER_REFLECT)

    # ========== 已删除：镜像 + 旋转组合 ==========

    # ========== 新增：亮度/对比度调整 ==========

    # 5. 增加亮度（+20%）
    brightness_up = np.random.uniform(1.15, 1.20)
    augmented_images['brightness_up'] = adjust_brightness_contrast(image, brightness_factor=brightness_up)
    if mask is not None:
        augmented_masks['brightness_up'] = mask.copy()

    # 6. 降低亮度（-20%）
    brightness_down = np.random.uniform(0.80, 0.85)
    augmented_images['brightness_down'] = adjust_brightness_contrast(image, brightness_factor=brightness_down)
    if mask is not None:
        augmented_masks['brightness_down'] = mask.copy()

    # 7. 增加对比度（+20%）
    contrast_up = np.random.uniform(1.15, 1.20)
    augmented_images['contrast_up'] = adjust_brightness_contrast(image, contrast_factor=contrast_up)
    if mask is not None:
        augmented_masks['contrast_up'] = mask.copy()

    # 8. 降低对比度（-20%）
    contrast_down = np.random.uniform(0.80, 0.85)
    augmented_images['contrast_down'] = adjust_brightness_contrast(image, contrast_factor=contrast_down)
    if mask is not None:
        augmented_masks['contrast_down'] = mask.copy()

    # ========== 已删除：亮度+对比度随机组合 ==========

    return (augmented_images, augmented_masks) if mask is not None else (augmented_images, None)


def augment_dataset_with_masks(image_dir, mask_dir, output_image_dir, output_mask_dir,
                               target_size=(512, 512)):
    """
    对图像和掩码进行批量数据增强，统一尺寸到512x512
    包含几何变换和亮度/对比度调整

    参数:
        image_dir: 输入图像文件夹路径
        mask_dir: 输入掩码文件夹路径
        output_image_dir: 输出图像文件夹路径
        output_mask_dir: 输出掩码文件夹路径
        target_size: 目标尺寸 (width, height)，默认 512x512
    """
    # 创建输出目录
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    # 支持的图像格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

    print(f"目标尺寸: {target_size}")
    print("增强方式: 几何变换 + 亮度/对比度调整")
    print("开始处理数据增强...\n")

    processed_count = 0
    total_augmented = 0

    # 遍历输入目录中的所有图像
    for img_path in sorted(Path(image_dir).iterdir()):
        if img_path.suffix.lower() not in image_extensions:
            continue

        # 读取原始图像
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"⚠ 无法读取图像: {img_path}")
            continue

        # 查找对应的掩码文件
        base_name = img_path.stem
        mask_path = None
        for ext in image_extensions:
            potential_mask = Path(mask_dir) / f"{base_name}{ext}"
            if potential_mask.exists():
                mask_path = potential_mask
                break

        if mask_path is None:
            print(f"⚠ 警告: 未找到对应的掩码文件: {base_name}")
            continue

        # 读取掩码图像
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            print(f"⚠ 无法读取掩码: {mask_path}")
            continue

        # 统一调整到目标尺寸 512x512
        if (image.shape[1], image.shape[0]) != target_size:
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

        print(f"✓ 处理: {img_path.name}")

        # 使用固定种子确保图像和掩码的随机变换一致
        seed = np.random.randint(0, 10000)

        # 对图像和掩码进行相同的增强
        augmented_images, augmented_masks = augment_image(image, mask, seed=seed)

        # 保存增强后的图像和掩码
        aug_count = 0
        for aug_type in augmented_images.keys():
            # 保存增强后的图像（使用高质量参数）
            img_output_name = f"{base_name}_{aug_type}{img_path.suffix}"
            img_output_path = os.path.join(output_image_dir, img_output_name)

            # 根据文件格式设置保存参数
            if img_path.suffix.lower() in ['.jpg', '.jpeg']:
                # JPEG格式：使用高质量（95-100）
                cv2.imwrite(img_output_path, augmented_images[aug_type],
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
            elif img_path.suffix.lower() == '.png':
                # PNG格式：使用最高压缩质量
                cv2.imwrite(img_output_path, augmented_images[aug_type],
                            [cv2.IMWRITE_PNG_COMPRESSION, 3])
            else:
                cv2.imwrite(img_output_path, augmented_images[aug_type])

            # 保存增强后的掩码（使用无损保存）
            mask_output_name = f"{base_name}_{aug_type}{mask_path.suffix}"
            mask_output_path = os.path.join(output_mask_dir, mask_output_name)

            if mask_path.suffix.lower() in ['.jpg', '.jpeg']:
                # 掩码用JPEG时也用高质量
                cv2.imwrite(mask_output_path, augmented_masks[aug_type],
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
            elif mask_path.suffix.lower() == '.png':
                # PNG掩码用无损压缩
                cv2.imwrite(mask_output_path, augmented_masks[aug_type],
                            [cv2.IMWRITE_PNG_COMPRESSION, 0])
            else:
                cv2.imwrite(mask_output_path, augmented_masks[aug_type])

            aug_count += 1

        print(f"  → 生成 {aug_count} 个增强版本\n")
        processed_count += 1
        total_augmented += aug_count

    print("=" * 60)
    print(f"✓ 数据增强完成!")
    print(f"  处理原始图像: {processed_count} 张")
    print(f"  生成增强图像: {total_augmented} 张")
    print(f"  增强倍数: {total_augmented / processed_count if processed_count > 0 else 0:.1f}x")
    print("=" * 60)


def augment_dataset_images_only(input_dir, output_dir, target_size=(512, 512)):
    """
    仅对图像进行批量增强(无掩码)，统一尺寸到512x512
    包含几何变换和亮度/对比度调整

    参数:
        input_dir: 输入图像文件夹路径
        output_dir: 输出图像文件夹路径
        target_size: 目标尺寸 (width, height)，默认 512x512
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 支持的图像格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

    print(f"目标尺寸: {target_size}")
    print("增强方式: 几何变换 + 亮度/对比度调整")
    print("开始处理数据增强...\n")

    processed_count = 0
    total_augmented = 0

    # 遍历输入目录中的所有图像
    for img_path in sorted(Path(input_dir).iterdir()):
        if img_path.suffix.lower() not in image_extensions:
            continue

        # 读取图像
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"⚠ 无法读取图像: {img_path}")
            continue

        # 统一调整到目标尺寸 512x512
        if (image.shape[1], image.shape[0]) != target_size:
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

        print(f"✓ 处理: {img_path.name}")

        # 进行数据增强
        augmented_images, _ = augment_image(image, mask=None)

        # 保存增强后的图像
        base_name = img_path.stem
        extension = img_path.suffix

        aug_count = 0
        for aug_type, aug_image in augmented_images.items():
            output_filename = f"{base_name}_{aug_type}{extension}"
            output_path = os.path.join(output_dir, output_filename)

            # 根据文件格式设置保存参数
            if extension.lower() in ['.jpg', '.jpeg']:
                cv2.imwrite(output_path, aug_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            elif extension.lower() == '.png':
                cv2.imwrite(output_path, aug_image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            else:
                cv2.imwrite(output_path, aug_image)

            aug_count += 1

        print(f"  → 生成 {aug_count} 个增强版本\n")
        processed_count += 1
        total_augmented += aug_count

    print("=" * 60)
    print(f"✓ 数据增强完成!")
    print(f"  处理原始图像: {processed_count} 张")
    print(f"  生成增强图像: {total_augmented} 张")
    print(f"  增强倍数: {total_augmented / processed_count if processed_count > 0 else 0:.1f}x")
    print("=" * 60)


# 使用示例
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Augment paired images and segmentation masks')
    parser.add_argument('--image-dir', required=True, help='Directory containing source images')
    parser.add_argument('--mask-dir', required=True, help='Directory containing source masks')
    parser.add_argument('--output-image-dir', required=True, help='Directory for augmented images')
    parser.add_argument('--output-mask-dir', required=True, help='Directory for augmented masks')
    parser.add_argument('--size', type=int, default=512, help='Square output size (default: 512)')
    args = parser.parse_args()

    # 图像+掩码批量增强，统一到512x512，包含几何变换和光照调整
    augment_dataset_with_masks(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        output_image_dir=args.output_image_dir,
        output_mask_dir=args.output_mask_dir,
        target_size=(args.size, args.size)
    )

    print("\n📋 增强方法说明:")
    print("=" * 60)
    print("🔷 几何变换 (4种):")
    print("✅ 1. original        - 原图（干净）")
    print("✅ 2. mirrored        - 水平镜像")
    print("✅ 3. rotated         - 随机旋转 (-5° ~ 5°)")
    print("✅ 4. perspective     - 透视变换（模拟摄像头角度变化）⭐")
    print("\n🔆 光照调整 (4种):")
    print("✅ 5. brightness_up   - 亮度增加 (+15~20%)")
    print("✅ 6. brightness_down - 亮度降低 (-15~20%)")
    print("✅ 7. contrast_up     - 对比度增加 (+15~20%)")
    print("✅ 8. contrast_down   - 对比度降低 (-15~20%)")
    print("=" * 60)
    print("\n💡 重要说明:")
    print("1. 所有图像统一resize到 512×512 尺寸")
    print("2. 每张原图会生成 8 个增强版本 (几何4+光照4)")
    print("3. 几何变换基于【干净的原图】")
    print("4. 光照调整模拟不同天气和光照条件:")
    print("   - 亮度调整: 模拟阴天/晴天/逆光")
    print("   - 对比度调整: 模拟雾天/清晰天气")
    print("5. 图像和掩码的几何变换完全一致")
    print("6. 光照调整只作用于图像，掩码保持不变")
    print("7. 透视变换专门优化用于车道线检测任务 ⭐")
    print("8. 确保原始图像和掩码文件名一致(扩展名可不同)")
    print("9. JPEG图像使用95%高质量保存，避免压缩噪点")
    print("10. PNG图像使用适度压缩，保持高质量")
    print("11. ❌ 已移除 mirror_rotate 和 light_adjust 组合增强")
