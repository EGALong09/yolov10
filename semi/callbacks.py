import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

import math
import os
import logging
from copy import deepcopy

class EMAUpdate:
    """在训练过程中更新EMA教师模型的回调。"""

    def __init__(self, teacher_model, decay=0.9996):
        """
        初始化EMA更新器。

        Args:
            teacher_model: 教师模型实例 (完整的YOLOv10对象)
            decay: EMA平滑系数。
        """
        # 【关键】我们操作的是YOLOv10对象内部的 nn.Module
        self.teacher_model_module = teacher_model.model
        self.decay = decay  # 【关键】使用从外部传入的decay值
        self.updates = 0
        self.logger = logging.getLogger(__name__)

    def __call__(self, trainer):
        # trainer.model 是学生模型的底层 nn.Module
        student_model_module = trainer.model

        with torch.no_grad():
            # 在训练的第一个step，直接将学生权重复制给教师，以确保完全同步
            if self.updates == 0:
                self.teacher_model_module = deepcopy(student_model_module)
                self.logger.info("EMA Teacher初始化完成，已与学生模型同步。")

            # 使用去偏（de-bias）的衰减率，这在训练早期更稳定
            self.updates += 1
            decay = self.decay * (1 - math.exp(-self.updates / 2000))

            student_sd = student_model_module.state_dict()
            teacher_sd = self.teacher_model_module.state_dict()

            for key in teacher_sd:
                if teacher_sd[key].is_floating_point():
                    teacher_sd[key].data.copy_(decay * teacher_sd[key].data + (1 - decay) * student_sd[key].data)


def update_thresholds_standalone(teacher_model, config, iteration):
    """
    一个独立的函数，用于在每次宏观迭代后更新FlexMatch动态阈值。

    Args:
        teacher_model (YOLOv10): 上一轮训练好的教师模型实例。
        config (dict): 完整的训练配置文件。
        iteration (int): 当前是第几轮宏观迭代。

    Returns:
        str: 生成的阈值文件的路径。
    """
    print(f"\n[FlexMatch] 开始为第 {iteration} 轮迭代更新动态阈值...")

    # 从config字典中安全地获取所有路径和参数
    unlabeled_data_path = config['unlabeled_images_dir']
    thresholds_dir = Path(config['generated_dataset_yamls_output_dir'])
    thresholds_dir.mkdir(parents=True, exist_ok=True)
    thresholds_path = thresholds_dir / f'flexmatch_thresholds_iter_{iteration}.json'

    num_classes = config.get('num_classes', teacher_model.model.nc)
    beta = config['flexmatch_config']['beta']
    sample_size = config['flexmatch_config']['threshold_update_sample_size']

    # ... (后续的采样、预测、计算和保存逻辑与之前完全相同) ...
    image_paths = list(Path(unlabeled_data_path).glob('*.*'))
    if not image_paths:
        print(f"警告: 在路径 {unlabeled_data_path} 未找到无标签图片，无法更新阈值。")
        return None

    sampled_paths = np.random.choice(image_paths, min(len(image_paths), sample_size), replace=False)
    confidences_by_class = [[] for _ in range(num_classes)]

    teacher_model.model.eval()
    with torch.no_grad():
        for img_path in tqdm(sampled_paths, desc="[FlexMatch] 临时抽样预测"):
            results = teacher_model(str(img_path), verbose=False, conf=0.01)
            for r in results:
                if hasattr(r, 'boxes') and r.boxes is not None:
                    for box in r.boxes:
                        cls, conf = int(box.cls[0]), float(box.conf[0])
                        if cls < num_classes:
                            confidences_by_class[cls].append(conf)

    prev_thresholds_path = thresholds_dir / f'flexmatch_thresholds_iter_{iteration - 1}.json'
    if prev_thresholds_path.exists():
        with open(prev_thresholds_path, 'r') as f:
            current_thresholds = json.load(f)
    else:
        current_thresholds = [config['flexmatch_config']['initial_threshold']] * num_classes

    for i in range(num_classes):
        if len(confidences_by_class[i]) > 10:
            percentile_value = np.percentile(confidences_by_class[i], beta * 100)
            current_thresholds[i] = round(float(percentile_value), 4)

    print(f"[FlexMatch] 第 {iteration} 轮迭代的新动态阈值为: {current_thresholds}")

    with open(thresholds_path, 'w') as f:
        json.dump(current_thresholds, f)
    print(f"[FlexMatch] 新阈值已保存至: {thresholds_path}")

    teacher_model.model.train()
    return str(thresholds_path)

def update_flexmatch_thresholds_on_epoch_end(trainer):
    """
    这是一个函数式回调，在每个epoch结束时被触发，用于更新动态阈值。
    它通过直接修改内存中数据集对象的属性来实现“在线”更新。
    """
    # 从trainer.args中安全地获取所有配置参数
    try:
        unlabeled_data_path = trainer.args.unlabeled_images_dir
        generated_config_dir = Path(trainer.args.generated_dataset_yamls_output_dir)
        num_classes = trainer.model.nc
        beta = trainer.args.beta
        warmup_epochs = trainer.args.warmup_epochs
        sample_size = trainer.args.threshold_update_sample_size
    except AttributeError as e:
        print(f"警告: [FlexMatch Callback] 无法从配置中获取必要参数，跳过更新。错误: {e}")
        return

    generated_config_dir.mkdir(parents=True, exist_ok=True)  # 确保目录存在
    thresholds_path = generated_config_dir / 'flexmatch_thresholds.json'

    # --- 1. 热身阶段判断 ---
    # 在训练的前几个epoch，模型不稳定，不应更新阈值
    if trainer.epoch < warmup_epochs:
        print(f"\n[FlexMatch Callback] 处于热身阶段 (Epoch {trainer.epoch + 1}/{warmup_epochs})，不更新阈值。")
        return

    # --- 2. 获取教师模型 ---
    teacher_model = None
    for callback_obj in trainer.callbacks.get('on_train_batch_end', []):
        if hasattr(callback_obj, 'teacher_model_module'):
            teacher_model = callback_obj.teacher_model_module
            break
    if teacher_model is None:
        teacher_model = trainer.model  # 如果没有EMA，则使用学生模型

    # 3. 采样并收集置信度
    image_paths = list(Path(unlabeled_data_path).glob('*.*'))
    if not image_paths: return

    sampled_paths = np.random.choice(image_paths, min(len(image_paths), sample_size), replace=False)
    confidences_by_class = [[] for _ in range(num_classes)]

    print(f"\n[FlexMatch Callback] 开始更新动态阈值 (采样 {len(sampled_paths)} 张图片)...")
    teacher_model.eval()
    with torch.no_grad():
        for img_path in tqdm(sampled_paths, desc="[FlexMatch Callback] 收集置信度"):
            results = teacher_model(str(img_path), verbose=False, conf=0.01)
            for r in results:
                if hasattr(r, 'boxes') and r.boxes is not None:
                    for box in r.boxes:
                        cls, conf = int(box.cls[0]), float(box.conf[0])
                        if cls < num_classes:
                            confidences_by_class[cls].append(conf)

        # 4. 计算新阈值
        with open(thresholds_path, 'r') as f:
            current_thresholds = json.load(f)

        for i in range(num_classes):
            if len(confidences_by_class[i]) > 10:
                percentile_value = np.percentile(confidences_by_class[i], beta * 100)
                current_thresholds[i] = round(float(percentile_value), 4)

        print(f"[FlexMatch Callback] 新的动态阈值为: {current_thresholds}")

        # 5. 保存到文件（用于持久化）并实时更新Dataset（实现在线课程学习的关键）
        with open(thresholds_path, 'w') as f:
            json.dump(current_thresholds, f)

        if hasattr(trainer.train_loader.dataset, 'dynamic_thresholds'):
            # 直接修改内存中数据集对象的阈值列表
            trainer.train_loader.dataset.dynamic_thresholds = current_thresholds
            print("[FlexMatch Callback] 已在运行时更新Dataset的阈值，将在下一个epoch生效。")

        teacher_model.train()