# Scallion Lane Segmentation

基于 PyTorch U-Net 的大葱田垄/车道线语义分割项目。模型面向田间导航场景，将图像分为背景、左侧车道线和右侧车道线，并提供训练、评估、批量推理与成对数据增强脚本。

本项目在 [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet) 基础上开发，增加了以下可选模块：

- **DSEM**：用于编码器的方向感知空间增强模块。
- **DACG**：用于解码器的方向感知上下文门控模块。
- **TGSM**：用于瓶颈层的拓扑引导空间模块，支持 `basic`、`enhanced` 和 `advanced` 三种版本。
- **边界感知损失**：支持交叉熵、边界损失和 focal-boundary 损失。
- **田间车道线指标**：按类别 IoU、mIoU、车道线 mIoU、MPA、Dice、Kappa 和 FPS。

> 数据集、训练权重、W&B 日志和推理结果不包含在仓库中。请只使用你有权公开或使用的数据。

## 环境

- Python 3.9+
- PyTorch 2.0+
- 推荐使用支持 CUDA 的 NVIDIA GPU

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

如需特定 CUDA 版本，请先按照 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/) 安装匹配的 `torch` 与 `torchvision`。

## 数据格式

默认目录结构：

```text
data/
├── images/
│   ├── sample_001.jpg
│   └── sample_002.jpg
└── masks/
    ├── sample_001.png
    └── sample_002.png
```

图像和掩码必须使用相同的文件名主体。当前训练配置将掩码像素值映射为：

| 原始像素值 | 训练类别 | 含义 |
| ---: | ---: | --- |
| 0 | 0 | Background |
| 85 | 1 | Left Lane |
| 170 | 2 | Right Lane |

## 数据增强

脚本会同步处理图像和掩码，包含镜像、轻微旋转、透视变换以及亮度/对比度变化：

```bash
python data_augmentation.py \
  --image-dir path/to/source/images \
  --mask-dir path/to/source/masks \
  --output-image-dir data/images \
  --output-mask-dir data/masks \
  --size 512
```

## 训练

基础 U-Net：

```bash
python train.py \
  --image-dir data/images \
  --mask-dir data/masks \
  --checkpoint-dir checkpoints \
  --classes 3 \
  --epochs 100 \
  --batch-size 4 \
  --amp
```

DSEM + DACG + TGSM 完整配置：

```bash
python train.py \
  --image-dir data/images \
  --mask-dir data/masks \
  --checkpoint-dir checkpoints \
  --classes 3 \
  --use-dsem \
  --use-dacg \
  --use-tgsm \
  --tgsm-version enhanced \
  --loss-type boundary \
  --amp
```

训练过程默认接入 Weights & Biases。首次使用时可运行 `wandb login`，也可以按 W&B 文档切换到离线模式。

## 评估

评估时必须使用与训练权重一致的模型开关：

```bash
python eval_fixed.py \
  --checkpoint checkpoints/checkpoint_epoch100.pth \
  --data-dir data/images \
  --mask-dir data/masks \
  --classes 3 \
  --use-dsem \
  --use-dacg \
  --use-tgsm \
  --tgsm-version enhanced
```

## 推理

单张或批量推理可使用 `inference_complete.py`：

```bash
# 单张图像
python inference_complete.py \
  --model checkpoints/checkpoint_epoch100.pth \
  --image path/to/image.jpg \
  --output output.png \
  --use-dsem --use-dacg --use-tgsm

# 目录批量推理
python inference_complete.py \
  --model checkpoints/checkpoint_epoch100.pth \
  --image-dir data/images \
  --mask-dir data/masks \
  --output-dir inference_results \
  --use-dsem --use-dacg --use-tgsm
```

## 项目结构

```text
├── train.py                       # 训练入口
├── eval_fixed.py                  # 完整评估入口
├── inference_complete.py          # 单张/批量推理
├── inference_compatible_fixed.py  # 旧权重兼容推理
├── data_augmentation.py           # 成对数据增强
├── unet/                          # U-Net、DSEM、DACG、TGSM
├── utils/                         # 数据加载、Dice、边界损失和指标
├── requirements.txt
└── Dockerfile
```

## 复现实验说明

- 仓库不包含 `.pth` 权重；模型文件通常较大，建议通过 GitHub Releases、对象存储或模型托管平台单独发布。
- 随机划分使用固定种子，但 GPU 算子、CUDA/cuDNN 版本仍可能带来细微差异。
- 使用自己的数据时，请确认掩码像素值与 `train.py` 中的 `original_labels` 一致。

## 许可与致谢

本项目继承上游项目的 [GNU General Public License v3.0](LICENSE)。发布修改版时请保留许可证、源代码和相应归属说明。

- 上游实现：[milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)
- 原始论文：Olaf Ronneberger, Philipp Fischer, Thomas Brox, [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

