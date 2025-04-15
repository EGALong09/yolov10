import os
import time
from pathlib import Path
from ultralytics import YOLOv10
import torch
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

def get_single_metric(value):
    """安全提取指标值（处理标量/单元素数组/多元素数组）"""
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.item()
        else:
            return value.mean()  # 多元素数组返回平均值
    return value

def validate():
    # 固定随机种子（确保可复现性）
    SEED = 42
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # 配置验证参数
    args = {
        'data': 'ultralytics/cfg/datasets/coco128.yaml',      # 数据集配置文件路径（与训练一致）
        'weights': '../runs/detect/train-v10n+coco128/weights/best.pt',  # 要验证的模型权重
        'batch': 32,                     # 批量大小（根据显存调整）
        'imgsz': 320,                    # 输入图像尺寸（与训练一致）
        'device': '0',                   # 设备（0为GPU，'cpu'为CPU）
        'workers': 4,                    # 数据加载线程数
        'conf': 0.001,                   # 置信度阈值（低阈值确保检出所有可能目标）
        'iou': 0.6,                      # NMS的IoU阈值
        'save_json': True,               # 保存结果为JSON（用于COCO指标计算）
        'save_hybrid': True,             # 保存混合标签（预测+真实框）
        'plots': True,                   # 生成评估曲线和混淆矩阵
        'name': 'val',               # 验证结果保存目录名
    }

    # 加载模型
    model = YOLOv10(args['weights'])

    # 手动构建保存目录（替代 model.val_dir）
    save_dir = Path("../runs") / "detect" / "log"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 记录开始时间
    start_time = time.time()
    print(f"\n{'=' * 30} 验证开始 {'=' * 30}")
    print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"模型: {args['weights']}")
    print(f"设备: {'GPU' if args['device'] != 'cpu' else 'CPU'}")

    # 开始验证
    metrics = model.val(
        data=args['data'],
        batch=args['batch'],
        imgsz=args['imgsz'],
        device=args['device'],
        workers=args['workers'],
        conf=args['conf'],
        iou=args['iou'],
        save_json=args['save_json'],
        save_hybrid=args['save_hybrid'],
        plots=args['plots'],
        name=args['name'],
        verbose=True  # 确保输出详细信息
    )

    # 计算FPS
    total_time = time.time() - start_time
    total_images = len(metrics.speed)
    avg_fps = total_images / total_time

    # 提取指标（安全处理各种返回类型）
    map50 = get_single_metric(metrics.box.map)
    map50_95 = get_single_metric(metrics.box.map50)
    precision = get_single_metric(metrics.box.p)
    recall = get_single_metric(metrics.box.r)

    # 准备要保存的结果文本
    result_text = f"""
    {'=' * 30} 验证结果 {'=' * 30}
    时间: {time.strftime('%Y-%m-%d %H:%M:%S')}
    模型: {args['weights']}
    设备: {'GPU' if args['device'] != 'cpu' else 'CPU'}

    关键指标:
    mAP@0.5: {map50:.4f}
    mAP@0.5:0.95: {map50_95:.4f}
    精确率: {precision:.4f}
    召回率: {recall:.4f}
    FPS: {avg_fps:.2f} (图片数: {total_images}, 耗时: {total_time:.2f}s)
    """

    # 如果有多个类别，添加类别详细指标
    if hasattr(metrics.box, 'maps') and isinstance(metrics.box.maps, np.ndarray):
        result_text += "\n各类别AP:\n"
        for i, ap in enumerate(metrics.box.maps):
            result_text += f"  类别 {i}: {ap:.4f}\n"

    result_text += f"{'=' * 60}\n"

    # 打印结果到控制台
    print(result_text)

    # 保存结果到文件
    log_file = save_dir / "val_results.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(result_text)

    print(f"验证完成！结果已保存到: {log_file}")
    return metrics


if __name__ == '__main__':
    validate()