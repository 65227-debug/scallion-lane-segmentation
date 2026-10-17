# Scallion Lane Segmentation

基于 PyTorch U-Net 的大葱田垄/车道线语义分割，输出背景、左侧车道线和右侧车道线。

## 环境

Python 3.10+、PyTorch 2.0+。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 模型

- DSEM：编码器中的方向感知空间增强模块。
- DACG：解码器中的方向感知上下文门控模块。
- TGSM：瓶颈层中的拓扑引导空间模块，支持 basic、enhanced、advanced。

```python
from unet import UNet

model = UNet(
    n_channels=3, n_classes=3,
    use_dsem=True, use_dacg=True, use_tgsm=True,
    tgsm_version='enhanced',
)
```

模型开关、上采样方式和 TGSM 版本须与权重对应。可先查看权重结构：

```bash
python detect_config.py model.pth
```

## 数据编码

图像使用 RGB；掩码使用单通道 PNG，图像与掩码文件名主体相同。

| 掩码像素值 | 类别索引 | 含义 |
| ---: | ---: | --- |
| 0 | 0 | 背景 |
| 85 | 1 | 左侧车道线 |
| 170 | 2 | 右侧车道线 |
| 255 | — | 忽略区域 |

## 推理

单张图像：

```bash
python inference_complete.py \
  --model model.pth \
  --image path/to/image.jpg \
  --output prediction.png \
  --use-dsem --use-dacg --use-tgsm \
  --tgsm-version enhanced
```

批量图像及配对掩码：

```bash
python inference_complete.py \
  --model model.pth \
  --image-dir path/to/images \
  --mask-dir path/to/masks \
  --output-dir inference_results \
  --use-dsem --use-dacg --use-tgsm \
  --tgsm-version enhanced
```

也可使用 `predict.py` 输出原始编码掩码：

```bash
python predict.py \
  --model model.pth \
  --input path/to/image.jpg \
  --output prediction.png \
  --classes 3 \
  --use-dsem --use-dacg --use-tgsm \
  --tgsm-version enhanced
```

旧权重的批量兼容入口为 `inference_compatible_fixed.py`，权重参数使用 `--model-path model.pth`。

## 评估

```bash
python eval_fixed.py \
  --checkpoint model.pth \
  --data-dir path/to/images \
  --mask-dir path/to/masks \
  --classes 3 \
  --use-dsem --use-dacg --use-tgsm \
  --tgsm-version enhanced
```

输出各类别 IoU、整体 mIoU、车道线 mIoU、MPA、Cohen's Kappa、参数量及推理速度。

## 许可与致谢

本项目基于 [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)，采用 [GNU GPL v3.0](LICENSE)。

U-Net：Olaf Ronneberger, Philipp Fischer, Thomas Brox, [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)。
