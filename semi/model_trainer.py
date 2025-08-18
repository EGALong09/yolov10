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
import functools
from pathlib import Path
import yaml
import json
from ultralytics import YOLOv10
from .callbacks import update_flexmatch_thresholds_on_epoch_end
import ultralytics.data.build as build
from .custom_dataset import FlexMatchInspiredDataset
from semi.callbacks import EMAUpdate, update_thresholds_standalone


class ModelTrainer:
    """模型训练器类"""

    def __init__(self, yolo_config):
        self.yolo_config = yolo_config
        self.logger = logging.getLogger(__name__)

    def train(self, student_model_instance, train_config, teacher_model_instance=None):
        """
        训练YOLOv10模型。
        这个方法现在只负责传递参数和调用训练，所有模式判断逻辑已移至 semi_train.py。
        """
        # 步骤 1: 从模型对象上安全地读取我们之前在 semi_train.py 中附加的自定义参数
        custom_args = getattr(student_model_instance, 'custom_args', {})

        # 步骤 2: 使用 functools.partial 创建一个“预设”了自定义参数的数据集类
        # 这样，当 ultralytics 内部调用它时，我们的参数会自动传入
        PatchedDataset = functools.partial(FlexMatchInspiredDataset, **custom_args)

        # 步骤 3: 进行猴子补丁，用我们预设好的类替换掉原始的类
        original_dataset_class = build.YOLODataset
        build.YOLODataset = PatchedDataset
        self.logger.info("已通过猴子补丁注入自定义参数到 FlexMatchInspiredDataset。")

        try:
            student_model = student_model_instance

            # 步骤 4: 合并所有配置项到一个字典中
            merged_args = self.yolo_config.copy()
            merged_args.update(train_config)

            # 步骤 5: 定义一个只包含 ultralytics 标准参数的“纯净白名单”
            standard_known_args = [
                'data', 'epochs', 'batch', 'imgsz', 'project', 'name', 'exist_ok', 'patience',
                'save', 'save_period', 'device', 'workers', 'amp', 'verbose', 'seed', 'lr0',
                'optimizer', 'momentum', 'weight_decay', 'warmup_epochs', 'close_mosaic',
                'hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'translate', 'scale', 'shear',
                'perspective', 'flipud', 'fliplr', 'mosaic', 'mixup', 'copy_paste', 'augment',
                'nwdloss', 'iou_ratio'
            ]

            # 步骤 6: 使用“纯净白名单”进行过滤，确保 final_train_args 不含任何自定义参数
            final_train_args = {key: val for key, val in merged_args.items() if key in standard_known_args}

            # 处理EMA和回调（这部分逻辑不变）
            if teacher_model_instance:
                # 检查必要的自定义参数是否存在
                if 'flexmatch_config' not in custom_args or 'unlabeled_images_dir' not in custom_args:
                    raise ValueError("半监督模式下，'flexmatch_config' 和 'unlabeled_images_dir' 必须被提供。")

                decay = merged_args.get('ema_decay', 0.9996)
                ema_updater = EMAUpdate(teacher_model_instance, decay=decay)
                student_model.add_callback("on_train_batch_end", ema_updater)
                student_model.add_callback("on_epoch_end", update_flexmatch_thresholds_on_epoch_end)

            self.logger.info(f"开始训练，传递给 model.train 的标准参数: {final_train_args}")

            # 步骤 7: 使用这个“纯净”的参数字典来调用训练，即可解决报错
            results = student_model.train(**final_train_args)

            save_dir = Path(train_config['project']) / train_config['name']
            best_model_path = save_dir / 'weights' / 'best.pt'
            if not best_model_path.exists():
                best_model_path = save_dir / 'weights' / 'last.pt'
            self.logger.info(f"训练完成，学生模型保存在: {best_model_path}")

            if teacher_model_instance:
                teacher_model_instance.save(save_dir / 'weights' / 'teacher_best.pt')
                self.logger.info(f"教师模型保存在: {save_dir / 'weights' / 'teacher_best.pt'}")
                if "on_train_batch_end" in student_model.callbacks:
                    student_model.callbacks["on_train_batch_end"] = []

            return str(best_model_path)

        except Exception as e:
            self.logger.error(f"训练过程中出错: {e}", exc_info=True)
            raise

        finally:
            build.YOLODataset = original_dataset_class
            self.logger.info("已恢复原始的YOLODataset类。")
            # 清理附加的属性，保持模型对象干净
            if hasattr(student_model_instance, 'custom_args'):
                del student_model_instance.custom_args

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