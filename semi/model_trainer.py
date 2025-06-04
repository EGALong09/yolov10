"""
模型训练器模块
封装YOLOv10模型的训练功能
"""

import os
import subprocess
import logging
from pathlib import Path
import yaml
import json


class ModelTrainer:
    """模型训练器类"""

    def __init__(self, config):
        """
        初始化模型训练器

        Args:
            config: YOLO训练配置
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

    def train(self, train_config):
        """
        训练YOLOv10模型

        Args:
            train_config: 训练配置字典，包含:
                - data: 数据集配置文件路径
                - model: 预训练模型或上一轮模型路径
                - epochs: 训练轮数
                - batch_size: 批处理大小
                - imgsz: 图像尺寸
                - project: 项目保存路径
                - name: 实验名称
                - lr0: 初始学习率（可选）
                - exist_ok: 是否覆盖已存在的实验（可选）

        Returns:
            训练好的模型权重路径
        """
        try:
            # 使用Ultralytics的Python API进行训练
            from ultralytics import YOLOv10

            # 加载模型
            model = YOLOv10(train_config['model'])

            # 准备训练参数
            train_args = {
                'data': train_config['data'],
                'epochs': train_config['epochs'],
                'batch': train_config['batch_size'],
                'imgsz': train_config['imgsz'],
                'project': train_config['project'],
                'name': train_config['name'],
                'exist_ok': train_config.get('exist_ok', False),
                'patience': self.config.get('patience', 50),
                'save': True,
                'save_period': -1,  # 只保存最佳和最后的模型
                'device': self.config.get('device', 0),  # GPU设备号
                'workers': self.config.get('workers', 8),
                'amp': self.config.get('amp', True),  # 自动混合精度
                'verbose': True,
                'seed': self.config.get('seed', 0)
            }

            # 如果指定了学习率，添加到参数中
            if 'lr0' in train_config:
                train_args['lr0'] = train_config['lr0']

            # 添加其他高级训练参数
            if 'optimizer' in self.config:
                train_args['optimizer'] = self.config['optimizer']
            if 'momentum' in self.config:
                train_args['momentum'] = self.config['momentum']
            if 'weight_decay' in self.config:
                train_args['weight_decay'] = self.config['weight_decay']
            if 'warmup_epochs' in self.config:
                train_args['warmup_epochs'] = self.config['warmup_epochs']
            if 'close_mosaic' in self.config:
                train_args['close_mosaic'] = self.config['close_mosaic']

            # 数据增强参数
            if 'augment' in self.config and self.config['augment']:
                augment_params = self.config.get('augment_params', {})
                for key, value in augment_params.items():
                    train_args[key] = value

            self.logger.info(f"开始训练，参数: {train_args}")

            # 执行训练
            results = model.train(**train_args)

            # 获取最佳模型路径
            save_dir = Path(train_config['project']) / train_config['name']
            best_model_path = save_dir / 'weights' / 'best.pt'

            if not best_model_path.exists():
                # 如果没有best.pt，使用last.pt
                best_model_path = save_dir / 'weights' / 'last.pt'

            self.logger.info(f"训练完成，模型保存在: {best_model_path}")

            # 保存训练结果摘要
            self._save_training_summary(save_dir, results)

            return str(best_model_path)

        except Exception as e:
            self.logger.error(f"训练过程中出错: {e}")
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