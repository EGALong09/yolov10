"""
模型训练器模块
封装YOLOv10模型的训练功能
"""
import math
import os
from copy import deepcopy

import torch
import subprocess
import logging
from pathlib import Path
import yaml
import json
from ultralytics import YOLOv10


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


class ModelTrainer:
    """模型训练器类"""

    def __init__(self, yolo_config):
        self.yolo_config = yolo_config
        self.logger = logging.getLogger(__name__)

    def train(self, student_model_instance, train_config, teacher_model_instance=None):
        """
        训练YOLOv10模型。
        直接接收并训练传入的模型对象。
        """
        try:
            # 【关键】直接使用传入的模型对象，不再重新创建
            student_model = student_model_instance

            # 如果传入了教师模型，则启用EMA
            if teacher_model_instance:
                decay = train_config.get('ema_decay', 0.9996)
                self.logger.info(f"启用EMA Teacher模式，decay={decay}")

                if student_model is teacher_model_instance:
                    raise ValueError("学生模型和教师模型不能是同一个对象实例！")

                # 创建回调实例，并传入教师模型实例
                ema_updater = EMAUpdate(teacher_model_instance, decay=decay)
                student_model.add_callback("on_train_batch_end", ema_updater)

            # 准备并净化传递给ultralytics的参数
            train_args = self.yolo_config.copy()
            train_args.update(train_config)

            known_args = [
                'data', 'epochs', 'batch', 'imgsz', 'project', 'name', 'exist_ok', 'patience',
                'save', 'save_period', 'device', 'workers', 'amp', 'verbose', 'seed', 'lr0',
                'optimizer', 'momentum', 'weight_decay', 'warmup_epochs', 'close_mosaic',
                'hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'translate', 'scale', 'shear',
                'perspective', 'flipud', 'fliplr', 'mosaic', 'mixup', 'copy_paste', 'augment'
            ]

            # 为了确保100%纯净，我们只挑选白名单里的参数
            final_train_args = {key: train_args[key] for key in known_args if key in train_args}

            self.logger.info(f"开始训练，最终传递参数: {final_train_args}")
            results = student_model.train(**final_train_args)

            save_dir = Path(train_config['project']) / train_config['name']
            best_model_path = save_dir / 'weights' / 'best.pt'
            if not best_model_path.exists():
                best_model_path = save_dir / 'weights' / 'last.pt'

            self.logger.info(f"训练完成，学生模型保存在: {best_model_path}")

            if teacher_model_instance:
                # 训练结束后保存教师模型权重（可选，但推荐）
                teacher_model_instance.save(save_dir / 'weights' / 'teacher_best.pt')
                self.logger.info(f"教师模型保存在: {save_dir / 'weights' / 'teacher_best.pt'}")
                if "on_train_batch_end" in student_model.callbacks:
                    self.logger.info("手动清除 on_train_batch_end 回调...")
                    student_model.callbacks["on_train_batch_end"] = []

            return str(best_model_path)

        except Exception as e:
            self.logger.error(f"训练过程中出错: {e}", exc_info=True)
            raise

    def train_with_cli(self, train_config):
        """
        使用命令行接口训练YOLOv10模型（备选方案）

        Args:
            train_config: 训练配置字典

        Returns:
            训练好的模型权重路径
        """
        # 构建命令行参数
        cmd = [
            'yolo', 'detect', 'train',
            f'data={train_config["data"]}',
            f'model={train_config["model"]}',
            f'epochs={train_config["epochs"]}',
            f'batch={train_config["batch_size"]}',
            f'imgsz={train_config["imgsz"]}',
            f'project={train_config["project"]}',
            f'name={train_config["name"]}'
        ]

        # 添加可选参数
        if 'lr0' in train_config:
            cmd.append(f'lr0={train_config["lr0"]}')
        if train_config.get('exist_ok', False):
            cmd.append('exist_ok=True')

        # 添加设备参数
        device = self.config.get('device', 0)
        cmd.append(f'device={device}')

        self.logger.info(f"执行训练命令: {' '.join(cmd)}")

        try:
            # 执行训练命令
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.logger.info("训练输出:")
            self.logger.info(result.stdout)

            # 获取模型路径
            save_dir = Path(train_config['project']) / train_config['name']
            best_model_path = save_dir / 'weights' / 'best.pt'

            return str(best_model_path)

        except subprocess.CalledProcessError as e:
            self.logger.error(f"训练命令执行失败: {e}")
            self.logger.error(f"错误输出: {e.stderr}")
            raise

    def _save_training_summary(self, save_dir, results):
        """
        保存训练结果摘要

        Args:
            save_dir: 保存目录
            results: 训练结果
        """
        try:
            summary = {
                'best_fitness': float(results.best_fitness) if hasattr(results, 'best_fitness') else None,
                'epochs_trained': len(results.results_dict) if hasattr(results, 'results_dict') else None,
                'final_metrics': {}
            }

            # 提取最终指标
            if hasattr(results, 'results_dict') and results.results_dict:
                last_epoch = list(results.results_dict.values())[-1]
                summary['final_metrics'] = {
                    'precision': float(last_epoch.get('metrics/precision(B)', 0)),
                    'recall': float(last_epoch.get('metrics/recall(B)', 0)),
                    'mAP50': float(last_epoch.get('metrics/mAP50(B)', 0)),
                    'mAP50-95': float(last_epoch.get('metrics/mAP50-95(B)', 0))
                }

            # 保存摘要
            summary_path = save_dir / 'training_summary.json'
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)

            self.logger.info(f"训练摘要已保存到: {summary_path}")

        except Exception as e:
            self.logger.warning(f"保存训练摘要时出错: {e}")

    def resume_training(self, checkpoint_path):
        """
        从检查点恢复训练

        Args:
            checkpoint_path: 检查点路径

        Returns:
            训练好的模型权重路径
        """
        try:
            from ultralytics import YOLO

            # 加载检查点
            model = YOLO(checkpoint_path)

            # 恢复训练
            results = model.train(resume=True)

            # 获取保存路径
            save_dir = Path(checkpoint_path).parent.parent
            best_model_path = save_dir / 'weights' / 'best.pt'

            return str(best_model_path)

        except Exception as e:
            self.logger.error(f"恢复训练失败: {e}")
            raise

    def fine_tune(self, base_model_path, dataset_config, fine_tune_config):
        """
        对模型进行微调

        Args:
            base_model_path: 基础模型路径
            dataset_config: 数据集配置
            fine_tune_config: 微调配置

        Returns:
            微调后的模型路径
        """
        # 准备微调参数
        train_config = {
            'data': dataset_config,
            'model': base_model_path,
            'epochs': fine_tune_config.get('epochs', 10),
            'batch_size': fine_tune_config.get('batch_size', 16),
            'imgsz': fine_tune_config.get('imgsz', 640),
            'project': fine_tune_config.get('project', 'runs/fine_tune'),
            'name': fine_tune_config.get('name', 'exp'),
            'lr0': fine_tune_config.get('lr0', 0.0001),  # 微调使用较小学习率
            'exist_ok': True
        }

        # 执行微调
        return self.train(train_config)