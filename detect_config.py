"""
自动检测Checkpoint所需的模型配置
通过分析参数名称和形状推断模型结构
"""

import torch
import argparse
from pathlib import Path


def detect_model_config(checkpoint_path):
    """
    自动检测checkpoint需要的模型配置
    """
    print("=" * 80)
    print("模型配置自动检测")
    print("=" * 80)
    print(f"分析checkpoint: {checkpoint_path}\n")

    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 提取state_dict
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # 过滤掉非tensor的值
    state_dict = {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}

    print(f"总参数数: {len(state_dict)}\n")

    # 检测配置
    config = {
        'n_channels': 3,
        'n_classes': 3,
        'bilinear': False,
        'use_dsem': False,
        'use_dacg': False,
        'use_tgsm': False,
        'tgsm_version': 'enhanced'
    }

    # 1. 检测bilinear
    # 查看down4的输出通道数
    down4_keys = [k for k in state_dict.keys() if 'down4' in k and 'weight' in k]
    if down4_keys:
        # 找到down4最后一层的weight
        for key in down4_keys:
            if 'conv2.0.weight' in key:  # DoubleConv的第二个卷积
                channels = state_dict[key].shape[0]
                print(f"检测到 down4 输出通道数: {channels}")
                if channels == 512:
                    config['bilinear'] = True
                    print("  → bilinear = True")
                elif channels == 1024:
                    config['bilinear'] = False
                    print("  → bilinear = False")
                break

    # 2. 检测DSEM
    dsem_keys = [k for k in state_dict.keys()
                 if 'dsem' in k.lower() or '.dir_convs.' in k]
    if dsem_keys:
        config['use_dsem'] = True
        print(f"\n检测到DSEM模块 ({len(dsem_keys)} 个参数)")
        print("  → use_dsem = True")

    # 3. 检测DACG
    dacg_keys = [k for k in state_dict.keys() if 'dacg' in k.lower() or 'gate_conv' in k]
    if dacg_keys:
        config['use_dacg'] = True
        print(f"\n检测到DACG模块 ({len(dacg_keys)} 个参数)")
        print("  → use_dacg = True")

    # 4. 检测TGSM
    tgsm_keys = [k for k in state_dict.keys() if 'tgsm' in k.lower() or 'tgb' in k.lower()]
    if tgsm_keys:
        config['use_tgsm'] = True
        print(f"\n检测到TGSM模块 ({len(tgsm_keys)} 个参数)")
        print("  → use_tgsm = True")

        if any('topo_gate.' in key for key in tgsm_keys):
            config['tgsm_version'] = 'advanced'
        elif any('dir_agg.proj_h.weight' in key for key in tgsm_keys):
            config['tgsm_version'] = 'basic'
        print(f"  → tgsm_version = '{config['tgsm_version']}'")

    # 5. 检测类别数
    outc_weight_keys = [k for k in state_dict.keys() if 'outc.conv.weight' in k]
    if outc_weight_keys:
        n_classes = state_dict[outc_weight_keys[0]].shape[0]
        config['n_classes'] = n_classes
        print(f"\n检测到输出类别数: {n_classes}")
        print(f"  → n_classes = {n_classes}")

    # 打印推荐的命令行参数
    print("\n" + "=" * 80)
    print("推荐的推理命令")
    print("=" * 80)

    cmd_parts = ["python inference_complete.py"]
    cmd_parts.append("    --image-dir <your_image_dir>")
    cmd_parts.append("    --mask-dir <your_mask_dir>")
    cmd_parts.append(f"    --model {checkpoint_path}")
    cmd_parts.append("    --output-dir ./inference_results")
    cmd_parts.append("    --scale 0.5")
    cmd_parts.append(f"    --classes {config['n_classes']}")

    if config['bilinear']:
        cmd_parts.append("    --bilinear")

    if config['use_dsem']:
        cmd_parts.append("    --use-dsem")

    if config['use_dacg']:
        cmd_parts.append("    --use-dacg")

    if config['use_tgsm']:
        cmd_parts.append("    --use-tgsm")
        cmd_parts.append(f"    --tgsm-version {config['tgsm_version']}")

    print("\n" + " \\\n".join(cmd_parts))

    # 打印Python代码
    print("\n" + "=" * 80)
    print("或者在Python代码中使用")
    print("=" * 80)
    print(f"""
model = UNet(
    n_channels={config['n_channels']},
    n_classes={config['n_classes']},
    bilinear={config['bilinear']},
    use_dsem={config['use_dsem']},
    use_dacg={config['use_dacg']},
    use_tgsm={config['use_tgsm']},
    tgsm_version='{config['tgsm_version']}'
)
""")

    print("=" * 80)
    print("\n✅ 配置检测完成！")
    print("\n⚠️  注意: 如果仍然加载失败，可能需要手动调整某些参数。")

    return config


def main():
    parser = argparse.ArgumentParser(description='Detect model architecture from a checkpoint')
    parser.add_argument('checkpoint_path', type=Path, help='Path to model.pth')
    args = parser.parse_args()
    if not args.checkpoint_path.is_file():
        parser.error(f'Checkpoint does not exist: {args.checkpoint_path}')
    detect_model_config(args.checkpoint_path)


if __name__ == '__main__':
    main()
