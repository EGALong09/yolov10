#!/usr/bin/env python3
"""
setup.py - YOLOv10半监督学习项目初始化脚本
"""

import os
import sys
from pathlib import Path
import argparse


def create_project_structure(project_root):
    """创建项目目录结构"""
    project_root = Path(project_root)
    
    directories = [
        # 数据目录
        'data/labeled/images/train',
        'data/labeled/labels/train',
        'data/labeled/images/val',
        'data/labeled/labels/val',
        'data/unlabeled/images',
        
        # 代码目录
        'scripts',
        'utils',
        'config',
        
        # 输出目录
        'runs',
        'logs',
    ]
    
    for dir_path in directories:
        (project_root / dir_path).mkdir(parents=True, exist_ok=True)
        
    print(f"✓ 项目目录结构已创建: {project_root}")
    

def create_init_files(project_root):
    """创建Python包初始化文件"""
    project_root = Path(project_root)
    
    # 创建__init__.py文件
    init_paths = [
        project_root / 'utils' / '__init__.py',
        project_root / 'scripts' / '__init__.py',
    ]
    
    for init_path in init_paths:
        init_path.touch(exist_ok=True)
        
    print("✓ Python包初始化文件已创建")
    

def create_requirements_file(project_root):
    """创建requirements.txt文件"""
    requirements = """# YOLOv10 半监督学习项目依赖

# YOLOv10 (Ultralytics)
ultralytics>=8.1.0

# 基础依赖
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
numpy>=1.24.0
Pillow>=10.0.0

# 数据处理
pandas>=2.0.0
scipy>=1.10.0
scikit-learn>=1.3.0

# 可视化
matplotlib>=3.7.0
seaborn>=0.12.0

# 工具库
tqdm>=4.65.0
PyYAML>=6.0
tensorboard>=2.13.0

# 可选：性能优化
# onnx>=1.14.0
# onnxruntime>=1.15.0
"""
    
    requirements_path = Path(project_root) / 'requirements.txt'
    with open(requirements_path, 'w') as f:
        f.write(requirements)
        
    print(f"✓ requirements.txt 已创建: {requirements_path}")
    

def create_readme(project_root):
    """创建README.md文件"""
    readme_content = """# YOLOv10 半监督学习训练项目

基于YOLOv10的交通目标检测半监督学习实现，通过迭代训练、伪标签生成和数据整合来提升模型性能。

## 项目结构

```
yolov10_ssl_project/
├── config/                 # 配置文件目录
│   └── train_config.yaml   # 训练配置文件
├── data/                   # 数据目录
│   ├── labeled/            # 有标签数据
│   │   ├── images/         # 图片
│   │   │   ├── train/      # 训练集图片
│   │   │   └── val/        # 验证集图片
│   │   └── labels/         # 标签
│   │       ├── train/      # 训练集标签
│   │       └── val/        # 验证集标签
│   └── unlabeled/          # 无标签数据
│       └── images/         # 无标签图片
├── scripts/                # 主要脚本
│   └── run_iterative_training.py  # 主训练脚本
├── utils/                  # 工具模块
│   ├── __init__.py
│   ├── pseudo_label_generator.py  # 伪标签生成器
│   ├── data_consolidator.py       # 数据整合器
│   ├── model_trainer.py           # 模型训练器
│   ├── model_evaluator.py         # 模型评估器
│   └── config_manager.py          # 配置管理器
├── runs/                   # 训练输出目录
├── logs/                   # 日志目录
└── requirements.txt        # 项目依赖

```

## 环境配置

### 1. 创建虚拟环境（推荐）

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\\Scripts\\activate  # Windows
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 下载YOLOv10预训练权重

```bash
# 下载YOLOv10n权重（最小模型，适合快速实验）
wget https://github.com/THU-MIG/yolov10/releases/download/v1.0/yolov10n.pt

# 或下载其他版本
# yolov10s.pt - 小型模型
# yolov10m.pt - 中型模型
# yolov10l.pt - 大型模型
# yolov10x.pt - 超大型模型
```

## 数据准备

### 1. 准备有标签数据

将少量高质量的标注数据放入对应目录：
- 训练图片：`data/labeled/images/train/`
- 训练标签：`data/labeled/labels/train/`
- 验证图片：`data/labeled/images/val/`
- 验证标签：`data/labeled/labels/val/`

标签格式为YOLO格式（每行一个目标）：
```
class_index x_center_norm y_center_norm width_norm height_norm
```

### 2. 准备无标签数据

将大量无标签图片放入：`data/unlabeled/images/`

### 3. 创建数据集配置文件

在 `data/` 目录下创建 `dataset_labeled.yaml`（参考提供的示例）。

## 配置文件说明

编辑 `config/train_config.yaml` 文件，主要配置项：

- `num_iterations`: 迭代轮数（建议3-5轮）
- `baseline_epochs`: 基准模型训练轮数
- `retrain_epochs`: 每轮迭代训练轮数
- `pseudo_label_conf_threshold`: 伪标签置信度阈值（建议0.3-0.5）
- `batch_size`: 根据GPU内存调整
- `device`: GPU设备号（使用CPU则设为'cpu'）

## 运行训练

### 基本训练命令

```bash
python scripts/run_iterative_training.py --config config/train_config.yaml
```

### 训练流程

1. **基准模型训练**：使用少量有标签数据训练初始模型
2. **迭代训练循环**：
   - 生成伪标签：使用当前模型对无标签数据预测
   - 数据整合：合并原始标注和伪标签数据
   - 模型再训练：使用合并数据训练新模型
   - 性能评估：在固定验证集上评估
   - 决定是否继续迭代

### 监控训练进程

训练日志保存在 `logs/` 目录，使用以下命令查看：
```bash
tail -f logs/training_*.log
```

## 结果查看

### 模型权重
- 基准模型：`runs/train_baseline/exp/weights/best.pt`
- 迭代模型：`runs/train_iter_N/exp/weights/best.pt`

### 评估结果
每个模型目录下包含：
- `evaluation_results.json`：评估指标
- `training_summary.json`：训练摘要

### 使用TensorBoard查看训练曲线
```bash
tensorboard --logdir runs/
```

## 模型推理

使用训练好的模型进行推理：

```python
from ultralytics import YOLO

# 加载模型
model = YOLO('runs/train_iter_3/exp/weights/best.pt')

# 推理
results = model('path/to/test/image.jpg')

# 显示结果
results[0].show()
```

## 常见问题

### 1. GPU内存不足
- 减小 `batch_size`
- 减小 `image_size`
- 使用更小的模型（如yolov10n）

### 2. 伪标签质量差
- 提高 `pseudo_label_conf_threshold`
- 增加基准模型训练轮数
- 确保有标签数据质量高

### 3. 性能不提升
- 检查无标签数据是否与目标域匹配
- 调整学习率和训练轮数
- 尝试不同的数据增强策略

## 进阶配置

### 自定义类别
编辑数据集配置文件中的 `names` 字段。

### 数据增强
在配置文件的 `augment_params` 中调整各项参数。

### 多GPU训练
设置 `device: [0,1,2,3]` 使用多个GPU。

## 许可证

本项目基于YOLOv10，遵循其开源协议。

## 致谢

- [YOLOv10](https://github.com/THU-MIG/yolov10)
- [Ultralytics](https://github.com/ultralytics/ultralytics)
"""
    
    readme_path = Path(project_root) / 'README.md'
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
        
    print(f"✓ README.md 已创建: {readme_path}")
    

def main():
    parser = argparse.ArgumentParser(description='初始化YOLOv10半监督学习项目')
    parser.add_argument('--project-root', type=str, default='./yolov10_ssl_project',
                        help='项目根目录路径')
    args = parser.parse_args()
    
    print("开始初始化YOLOv10半监督学习项目...")
    print("="*50)
    
    # 创建项目结构
    create_project_structure(args.project_root)
    
    # 创建初始化文件
    create_init_files(args.project_root)
    
    # 创建requirements.txt
    create_requirements_file(args.project_root)
    
    # 创建README.md
    create_readme(args.project_root)
    
    print("="*50)
    print("✓ 项目初始化完成！")
    print(f"\n下一步：")
    print(f"1. cd {args.project_root}")
    print(f"2. 将代码文件复制到对应目录")
    print(f"3. 准备数据集")
    print(f"4. 配置 config/train_config.yaml")
    print(f"5. 运行 python scripts/run_iterative_training.py")
    

if __name__ == '__main__':
    main()