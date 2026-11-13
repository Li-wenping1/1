# MobileNetV4 多任务视觉检测与病害区域分析

本项目实现了一套面向图像病害/缺陷分析的多任务视觉 pipeline：

1. 使用 YOLO 检测头完成目标检测；
2. 使用并行的 `20 × 20` 二值网格头预测图像中的病害区域；
3. 使用网格结果筛选候选区域；
4. 可选地调用 DeepLabV3+ 对候选区域进行局部分割；
5. 输出检测框、网格热力图、分割掩膜以及 PCI 风格的病害密度指标。

项目的多任务实现位于 [`multitask/`](D:/MobileNetV4/multitask)，YOLO 代码位于 [`yolov12/`](D:/MobileNetV4/yolov12)，DeepLabV3+ 代码位于 [`deeplabv3-plus-pytorch-main/`](D:/MobileNetV4/deeplabv3-plus-pytorch-main)。

> 当前脚本中的默认路径是 Windows 本地绝对路径。将项目复制到其他机器后，需要修改脚本顶部的配置，或通过命令行参数覆盖。

## 功能特点

- 基于 Ultralytics YOLO `DetectionModel` 封装多任务模型，不修改原始 YOLO 源码；
- 检测头和网格头共享 YOLO backbone 特征；
- 支持 `multitask`、`grid`、`detection` 三种训练模式；
- 支持 BCE、Focal BCE、正样本权重和损失 EMA 归一化；
- 支持 LetterBox、HSV、平移、缩放、水平翻转和 Mosaic 增强；
- 验证阶段计算网格准确率、Precision、Recall、F1、AP 和检测 mAP50；
- 自动保存 last、best loss、best grid F1、best mAP50 权重；
- 支持批量测试、结果图片、预测 CSV 和指标 CSV 输出。

## 项目结构

```text
MobileNetV4/
├── multitask/
│   ├── model_multitask.py       # YOLO 检测头 + 网格预测头
│   ├── loss_multitask.py        # 检测损失与网格损失
│   ├── dataset_multitask.py     # 数据读取、LetterBox、增强和 collate
│   ├── generate_grid_labels.py  # 从 YOLO 标签生成网格标签
│   ├── train_multitask.py       # 训练和验证
│   ├── test_visualize.py        # 批量测试和可视化
│   ├── post_process.py           # NMS、IoA 和网格后处理
│   └── pipeline_infer.py        # 检测、网格筛选、DeepLab 分割 pipeline
├── yolov12/                     # 本地 Ultralytics YOLO 工程
├── deeplabv3-plus-pytorch-main/ # DeepLabV3+ 工程
└── README.md
```

## 环境安装

建议使用 Python 3.9+、PyTorch 以及 CUDA 环境。先安装 PyTorch，再安装项目依赖：

```bash
pip install numpy pillow tqdm matplotlib
pip install -e ./yolov12
```

如果要使用 DeepLabV3+，还需要根据 [`deeplabv3-plus-pytorch-main/requirements.txt`](D:/MobileNetV4/deeplabv3-plus-pytorch-main/requirements.txt) 安装对应依赖。不同 CUDA 版本的 PyTorch 安装命令请以 PyTorch 官网为准。

## 数据格式

训练集和验证集分别采用以下目录结构：

```text
dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── grids/
    ├── train/
    └── val/
```

三类文件通过**相同文件名 stem** 对齐：

```text
images/train/0001.jpg
labels/train/0001.txt
grids/train/0001.npy
```

### YOLO 标签

每行一个目标，格式为：

```text
class_id center_x center_y width height
```

坐标为相对于图像宽高归一化到 `[0, 1]` 的值，例如：

```text
0 0.512 0.438 0.220 0.175
```

### 网格标签

网格标签是 NumPy 文件，形状必须为：

```text
(20, 20)
```

值为 `0/1`。默认情况下，目标框只要与网格单元有面积交集，该单元就会被标记为正样本。

## 生成网格标签

可以根据已有 YOLO `.txt` 标签生成 `.npy` 网格标签：

```bash
python -m multitask.generate_grid_labels ^
  --labels D:\path\to\dataset\labels\train ^
  --grids D:\path\to\dataset\grids\train ^
  --grid-size 20 ^
  --min-ioa 0
```

上面命令使用 Windows CMD 的 `^` 换行；Linux/macOS 请改用 `\`，或将命令写成一行。`--min-ioa` 表示目标框与网格单元交集面积占网格单元面积的最小比例：

- `0`：只要接触到网格单元就标记；
- 更大的值：只保留覆盖更充分的网格单元。

## 训练

训练前至少确认以下路径正确：

- 训练图片、YOLO 标签、网格标签；
- 验证图片、YOLO 标签、网格标签；
- YOLO 模型配置文件；
- `yolov12` 工程目录。

最简单的启动方式：

```bash
python -m multitask.train_multitask
```

也可以通过参数覆盖默认配置：

```bash
python -m multitask.train_multitask ^
  --images D:\path\to\dataset\images\train ^
  --labels D:\path\to\dataset\labels\train ^
  --grids D:\path\to\dataset\grids\train ^
  --val-images D:\path\to\dataset\images\val ^
  --val-labels D:\path\to\dataset\labels\val ^
  --val-grids D:\path\to\dataset\grids\val ^
  --model-cfg D:\MobileNetV4\yolov12\ultralytics\cfg\models\12\yolo12.yaml ^
  --yolo-root D:\MobileNetV4\yolov12 ^
  --output D:\path\to\weights\multitask.pt ^
  --task-mode multitask ^
  --num-classes 2 ^
  --image-size 640 ^
  --grid-size 20 ^
  --batch-size 4 ^
  --epochs 100 ^
  --device cuda
```

### 训练模式

| 模式 | 说明 |
|---|---|
| `multitask` | 同时训练 YOLO 检测头和网格头 |
| `grid` | 训练/评估网格预测任务 |
| `detection` | 训练/评估目标检测任务 |

常用参数：

```bash
--optimizer AdamW
--learning-rate 0.0001
--grid-loss bce
--grid-loss focal --focal-gamma 2
--grid-threshold 0.7
--use-augmentation
--no-augmentation
--use-cosine-scheduler
--no-cosine-scheduler
```

训练输出包括：

- `*_last.pt`
- `*_best_loss.pt`
- `*_best_f1.pt`
- `*_best_map50.pt`
- `metrics.csv`
- `map50_curve.png`

## 批量测试与可视化

编辑 [`multitask/test_visualize.py`](D:/MobileNetV4/multitask/test_visualize.py) 顶部的测试图片、标签、网格标签、权重和模型配置路径，然后运行：

```bash
python -m multitask.test_visualize
```

默认会在 `OUTPUT_DIR` 下生成：

- `*_result.jpg`：网格热力图和检测框可视化；
- `predictions.csv`：每张图片的网格统计和检测框；
- `test_metrics.csv`：网格指标与检测 mAP50。

## 端到端推理

[`multitask/pipeline_infer.py`](D:/MobileNetV4/multitask/pipeline_infer.py) 提供 `InferencePipeline`，将多任务检测、网格筛选和 DeepLabV3+ 局部分割串联起来：

```python
from multitask.pipeline_infer import InferencePipeline, load_deeplab

deeplab = load_deeplab(
    project_dir=r"D:\MobileNetV4\deeplabv3-plus-pytorch-main",
    weights=r"D:\path\to\deeplab_weights.pth",
    num_classes=2,
    device="cuda",
)

pipeline = InferencePipeline(
    multitask_model=multitask_model,
    deeplab_model=deeplab,
    device="cuda",
    grid_threshold=0.5,
    ioa_threshold=0.1,
)

result = pipeline(r"D:\path\to\image.jpg")
mask = result["mask"]
pci = result["pci"]
```

返回结果包含：

```python
{
    "grid": ...,   # 恢复到原图坐标的网格概率
    "cells": ...,  # 通过阈值和 IoA 筛选的网格单元
    "mask": ...,   # DeepLab 局部分割合并后的掩膜
    "pci": ...,    # 基于网格病害比例的 PCI proxy
}
```

## 训练流程概览

```text
YOLO 图片 + YOLO 框标签
          │
          ├── 生成 20×20 网格标签
          │
          ▼
    MultiTaskYOLO
      ├── YOLO Detection Head ──> 检测框
      └── Grid Head             ──> 20×20 网格概率
                                      │
                                      ▼
                              阈值 + IoA 筛选候选区域
                                      │
                                      ▼
                              DeepLabV3+ 局部分割
                                      │
                                      ▼
                         掩膜、可视化结果、PCI proxy
```

## 注意事项

- `NUM_CLASSES` 必须与 YOLO 标签中的类别数一致；
- 每张图片都必须有对应的 `.npy` 网格标签，否则数据集会报错；
- `grid_size` 训练、标签生成和推理配置必须一致；
- 训练和测试使用相同的图像尺寸时，默认输入尺寸为 `640 × 640`；
- 推理阶段的检测框需要经过 LetterBox 逆变换后才映射回原图；
- PCI 输出是基于网格病害比例计算的 proxy，不应直接视为经过标定的工程 PCI 指标；
- 使用本地 Ultralytics 代码时，请同时遵守其许可证和第三方模型权重的许可证。

## 相关模块

- [`model_multitask.py`](D:/MobileNetV4/multitask/model_multitask.py)：多任务模型结构
- [`loss_multitask.py`](D:/MobileNetV4/multitask/loss_multitask.py)：联合损失
- [`dataset_multitask.py`](D:/MobileNetV4/multitask/dataset_multitask.py)：数据集和增强
- [`generate_grid_labels.py`](D:/MobileNetV4/multitask/generate_grid_labels.py)：网格标签生成
- [`train_multitask.py`](D:/MobileNetV4/multitask/train_multitask.py)：训练、验证和指标记录
- [`test_visualize.py`](D:/MobileNetV4/multitask/test_visualize.py)：测试和结果可视化
- [`post_process.py`](D:/MobileNetV4/multitask/post_process.py)：后处理
- [`pipeline_infer.py`](D:/MobileNetV4/multitask/pipeline_infer.py)：端到端推理


