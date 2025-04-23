import os
from ultralytics import YOLOv10

# 固定随机种子（确保可复现性）
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
import random
import numpy as np
import torch
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

def train():
    # 配置训练参数
    args = {
        'data': 'ultralytics/cfg/datasets/tt100k.yaml',      # 数据集配置文件路径
        'weights': 'yolov10n.pt',        # 初始权重（官方预训练或None从头训练）
        'epochs': 100,                   # 训练轮次
        'batch': 32,                     # 批量大小
        'imgsz': 320,                    # 输入图像尺寸
        'device': '0',                   # 设备（0为GPU，'cpu'为CPU）
        'workers': 4,                    # 数据加载线程数
        # 'optimizer': 'auto',             # 优化器（auto/SGD/Adam/AdamW）
        'lr0': 0.01,                     # 初始学习率
        # 'cos_lr': True,                  # 使用余弦退火学习率调度
        'label_smoothing': 0.1,          # 标签平滑系数
        'patience': 50,                  # 早停耐心值（epochs无改善后停止）
        'save': True,                    # 保存模型
        'exist_ok': False,               # 是否覆盖已有实验
        'pretrained': True,              # 是否使用预训练权重
        'verbose': True,                 # 打印详细日志
        'seed': SEED,                    # 随机种子
    }

    # 初始化模型
    model = YOLOv10(args['weights'])

    # 开始训练
    results = model.train(
        data=args['data'],
        epochs=args['epochs'],
        batch=args['batch'],
        imgsz=args['imgsz'],
        device=args['device'],
        workers=args['workers'],
        # optimizer=args['optimizer'],
        lr0=args['lr0'],
        # cos_lr=args['cos_lr'],
        label_smoothing=args['label_smoothing'],
        patience=args['patience'],
        save=args['save'],
        exist_ok=args['exist_ok'],
        pretrained=args['pretrained'],
        verbose=args['verbose'],
        seed=args['seed'],
    )

    # 返回最终mAP
    print(f"训练完成，最终mAP50-95: {results.results_dict['metrics/mAP50-95(B)']:.3f}")

if __name__ == '__main__':
    train()