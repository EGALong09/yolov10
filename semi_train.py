#!/usr/bin/env python3
"""
基于YOLOv10的半监督迭代训练主控制脚本
用于管理整个迭代训练流程，包括基准模型训练、伪标签生成、数据整合和模型再训练
"""

import os
import sys
import yaml
import argparse
import logging
from pathlib import Path
from datetime import datetime

# 导入工具模块
from semi.pseudo_label_generator import PseudoLabelGenerator
from semi.data_consolidator import DataConsolidator
from semi.model_trainer import ModelTrainer
from semi.model_evaluator import ModelEvaluator
from semi.config_manager import ConfigManager


class IterativeTrainer:
    """半监督迭代训练控制器"""

    def __init__(self, config_path):
        """
        初始化迭代训练器

        Args:
            config_path: 配置文件路径
        """
        self.raw_config = ConfigManager.load_config(config_path)  # 先加载原始配置
        self.config = self._resolve_paths(self.raw_config)  # 解析路径
        self.setup_logging()
        # self.setup_directories() # 现在目录主要在 datasets_base_dir，按需创建

        # 初始化各个组件
        self.trainer = ModelTrainer(self.config['yolo_config'])
        # 将完整config或需要的路径部分传递给其他组件
        self.pseudo_generator = PseudoLabelGenerator(self.config)  # 或者只传 pseudo_label_config 和解析后的路径
        self.data_consolidator = DataConsolidator(self.config)  # 或者只传 data_config 和解析后的路径
        self.evaluator = ModelEvaluator(self.config['eval_config'])

        # 训练状态
        self.current_iteration = 0
        self.best_mAP = 0.0
        self.model_weights_history = []

    def _resolve_paths(self, config):
        """解析配置文件中的路径，将相对路径转换为基于 datasets_base_dir 的绝对路径"""
        resolved_config = config.copy()
        base_dir = Path(config['datasets_base_dir']).resolve()  # 获取绝对路径

        path_keys_to_resolve = [
            'initial_dataset_yaml',
            'labeled_data_root_dir',
            'unlabeled_images_dir',
            'pseudo_labels_output_root',
            'consolidated_data_output_root',
            'generated_dataset_yamls_output_dir',
            'fixed_validation_images_path',
            'fixed_validation_labels_path',  # 如果使用
        ]

        for key in path_keys_to_resolve:
            if key in resolved_config:
                path_val = Path(resolved_config[key])
                if not path_val.is_absolute():
                    resolved_config[key] = str(base_dir / path_val)
                else:
                    resolved_config[key] = str(path_val)  # 保持绝对路径

        # project_root 用于项目自身文件，保持原样或也设为绝对路径
        resolved_config['project_root'] = str(Path(config['project_root']).resolve())

        # 确保传递给YOLO的data路径是它能理解的
        # initial_dataset_yaml 已经是绝对路径了，YOLO可以直接使用

        # 为各个组件准备好它们需要的路径
        resolved_config['yolo_config']['project_root_for_yolo_runs'] = str(
            Path(resolved_config['project_root']) / 'runs')

        # 无标签图片目录，确保是字符串
        resolved_config['unlabeled_images_dir_str'] = str(Path(resolved_config['unlabeled_images_dir']))
        # 初始有标签数据集的YAML路径
        resolved_config['initial_dataset_yaml_str'] = str(Path(resolved_config['initial_dataset_yaml']))

        # 固定的验证集图片路径 (给 DataConsolidator 用来写入新YAML)
        resolved_config['fixed_validation_images_path_str'] = str(Path(resolved_config['fixed_validation_images_path']))
        if 'fixed_validation_labels_path' in resolved_config and resolved_config['fixed_validation_labels_path']:
            resolved_config['fixed_validation_labels_path_str'] = str(
                Path(resolved_config['fixed_validation_labels_path']))
        else:
            resolved_config['fixed_validation_labels_path_str'] = None

        return resolved_config

    def setup_logging(self):
        """设置日志系统"""
        log_dir = Path(self.config['project_root']) / 'semi_logs'
        log_dir.mkdir(exist_ok=True)

        log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    # def setup_directories(self): # 这个方法可能需要调整或移除，因为主要目录在外部
    #     """创建项目本身必要的目录结构 (如 project_root/runs, logs)"""
    #     project_root_path = Path(self.config['project_root'])
    #     (project_root_path / 'runs').mkdir(parents=True, exist_ok=True)
    #     (project_root_path / 'logs').mkdir(parents=True, exist_ok=True)
    #     # datasets_base_dir 及其子目录由用户或各模块按需创建
    #     self.logger.info("项目运行和日志目录已准备。")

    def setup_directories(self):
        """创建必要的目录结构"""
        project_root = Path(self.config['project_root'])

        # 创建所需的目录
        directories = [
            'data/labeled/images/train',
            'data/labeled/labels/train',
            'data/labeled/images/val',
            'data/labeled/labels/val',
            'data/unlabeled/images',
            'runs',
            'scripts',
            'semi_logs'
        ]

        for dir_path in directories:
            (project_root / dir_path).mkdir(parents=True, exist_ok=True)

        self.logger.info("目录结构已创建完成")

    def train_baseline_model(self):
        """
        阶段1: 训练初始基准模型
        使用少量有标签数据训练第一个YOLOv10模型
        """
        self.logger.info("=" * 50)
        self.logger.info("开始训练基准模型")

        project_for_yolo_runs = Path(self.config['project_root']) / 'runs'  # 训练输出仍在项目内

        # 配置基准模型训练参数
        baseline_config = {
            'data': self.config['initial_dataset_yaml_str'], # 使用解析后的绝对路径
            'model': self.config['pretrained_model'],
            'epochs': self.config['baseline_epochs'],
            'batch_size': self.config['batch_size'],
            'imgsz': self.config['image_size'],
            'project': str(project_for_yolo_runs / 'train_baseline'), # 路径调整
            'name': 'exp',
            'exist_ok': True
        }

        # 执行训练
        model_path = self.trainer.train(baseline_config)
        self.model_weights_history.append(model_path)

        # 评估基准模型
        metrics = self.evaluator.evaluate(
            model_path,
            self.config['initial_dataset_yaml_str']
        )

        self.logger.info(f"基准模型训练完成，mAP@0.5: {metrics['mAP50']:.4f}")
        self.best_mAP = metrics['mAP50']

        return model_path

    def generate_pseudo_labels(self, model_path, iteration):
        """
        阶段2: 生成伪标签
        使用当前模型对无标签数据进行预测，生成伪标签

        Args:
            model_path: 模型权重路径
            iteration: 当前迭代轮数
        """
        self.logger.info(f"开始生成第{iteration}轮伪标签")

        # 设置伪标签输出目录
        # 使用配置中 pseudo_labels_output_root
        pseudo_labels_iter_root = Path(self.config['pseudo_labels_output_root']) / f'iter_{iteration}'
        pseudo_labels_dir = pseudo_labels_iter_root / 'labels'  # 伪标签存在labels子文件夹
        pseudo_labels_dir.mkdir(parents=True, exist_ok=True)

        # 生成伪标签
        pseudo_label_config = {
            'weights': model_path,
            'source_images': self.config['unlabeled_images_dir_str'],
            'output_dir': str(pseudo_labels_dir),
            'conf_threshold': self.config['pseudo_label_conf_threshold'],
            'img_size': self.config['image_size']
        }

        num_generated = self.pseudo_generator.generate(pseudo_label_config)
        self.logger.info(f"生成了{num_generated}个伪标签文件")

        return str(pseudo_labels_dir)

    def consolidate_data(self, pseudo_labels_dir_str, iteration):
        """
        阶段3: 数据整合
        将原始有标签数据与新生成的伪标签数据合并

        Args:
            pseudo_labels_dir: 伪标签目录
            iteration: 当前迭代轮数
        """
        self.logger.info(f"开始整合第{iteration}轮数据")

        # 使用配置中的 consolidated_data_output_root 和 generated_dataset_yamls_output_dir
        consolidated_iter_dir = Path(self.config['consolidated_data_output_root']) / f'iter_{iteration}'
        # consolidated_iter_dir.mkdir(parents=True, exist_ok=True) # DataConsolidator会创建

        dataset_yaml_iter_path = Path(self.config['generated_dataset_yamls_output_dir']) / f'dataset_iter_{iteration}.yaml'
        dataset_yaml_iter_path.parent.mkdir(parents=True, exist_ok=True)

        # 执行数据整合
        consolidate_config = {
            'original_labeled_data_root_dir': self.config['labeled_data_root_dir'],  # 解析后的有标签数据根目录
            'unlabeled_images_input_dir': self.config['unlabeled_images_dir_str'],  # 解析后的无标签图片目录
            'pseudo_labels_input_dir': pseudo_labels_dir_str,  # 上一步生成的伪标签目录
            'consolidated_output_dir': str(consolidated_iter_dir),  # 合并后数据存放目录
            'new_dataset_yaml_path': str(dataset_yaml_iter_path),  # 新生成的YAML文件路径
            'fixed_validation_images_path': self.config['fixed_validation_images_path_str'],
            'fixed_validation_labels_path': self.config.get('fixed_validation_labels_path_str'),  # 可选
            'original_dataset_yaml_for_names': self.config['initial_dataset_yaml_str']  # 用于获取类别名
        }

        dataset_yaml_output = self.data_consolidator.consolidate(consolidate_config)
        self.logger.info(f"数据整合完成，新数据集配置文件: {dataset_yaml_output}")
        return dataset_yaml_output

    def retrain_model(self, dataset_yaml, previous_model_path, iteration):
        """
        阶段4: 模型再训练
        使用合并后的数据集训练新的YOLOv10模型

        Args:
            dataset_yaml: 数据集配置文件路径
            previous_model_path: 上一轮模型权重路径
            iteration: 当前迭代轮数
        """
        self.logger.info(f"开始第{iteration}轮模型再训练")

        project_for_yolo_runs = Path(self.config['project_root']) / 'runs'  # 训练输出仍在项目内

        # 配置再训练参数
        retrain_config = {
            'data': dataset_yaml,  # 这是新生成的、指向外部数据的YAML绝对路径
            'model': previous_model_path,
            'epochs': self.config['retrain_epochs'],
            'batch_size': self.config['batch_size'],
            'imgsz': self.config['image_size'],
            'project': str(project_for_yolo_runs / f'train_iter_{iteration}'),  # 路径调整
            'name': 'exp',
            'lr0': self.config['retrain_learning_rate'],
            'exist_ok': True
        }

        # 执行训练
        model_path = self.trainer.train(retrain_config)
        self.model_weights_history.append(model_path)

        return model_path

    def evaluate_and_decide(self, model_path, iteration):
        """
        阶段5: 评估模型并决定是否继续迭代

        Args:
            model_path: 模型权重路径
            iteration: 当前迭代轮数

        Returns:
            bool: 是否继续迭代
        """
        self.logger.info(f"评估第{iteration}轮模型性能")

        # 在固定验证集上评估
        metrics = self.evaluator.evaluate(
            model_path,
            self.config['initial_dataset_yaml_str']
        )

        current_mAP = metrics['mAP50']
        self.logger.info(f"第{iteration}轮 mAP@0.5: {current_mAP:.4f}")

        # 计算性能提升
        improvement = current_mAP - self.best_mAP
        self.logger.info(f"相比最佳模型提升: {improvement:.4f}")

        # 决定是否继续迭代
        if improvement > self.config['min_improvement_threshold']:
            self.best_mAP = current_mAP
            self.logger.info("性能提升明显，继续迭代")
            return True
        else:
            self.logger.info("性能提升不明显，考虑停止迭代")
            return improvement > 0  # 如果还有提升就继续，否则停止

    def train(self):
        """
        主训练函数：执行完整的半监督迭代训练流程
        """
        self.logger.info("开始半监督迭代训练流程")
        self.logger.info(f"总迭代轮数: {self.config['num_iterations']}")

        # 阶段0: 训练基准模型
        baseline_model = self.train_baseline_model()

        # 迭代训练循环
        current_model = baseline_model

        for iteration in range(1, self.config['num_iterations'] + 1):
            self.current_iteration = iteration
            self.logger.info(f"\n{'=' * 50}")
            self.logger.info(f"开始第 {iteration}/{self.config['num_iterations']} 轮迭代")

            try:
                # 阶段2: 生成伪标签
                pseudo_labels_dir = self.generate_pseudo_labels(current_model, iteration)

                # 阶段3: 数据整合
                dataset_yaml = self.consolidate_data(pseudo_labels_dir, iteration)

                # 阶段4: 模型再训练
                new_model = self.retrain_model(dataset_yaml, current_model, iteration)

                # 阶段5: 评估并决定是否继续
                should_continue = self.evaluate_and_decide(new_model, iteration)

                # 更新当前模型
                current_model = new_model

                # 如果性能不再提升，提前停止
                if not should_continue and self.config['early_stopping']:
                    self.logger.info("触发早停机制，结束训练")
                    break

            except Exception as e:
                self.logger.error(f"第{iteration}轮迭代出错: {str(e)}")
                if self.config['continue_on_error']:
                    self.logger.info("继续下一轮迭代")
                    continue
                else:
                    raise

        # 训练完成，输出总结
        self.logger.info("\n" + "=" * 50)
        self.logger.info("半监督迭代训练完成！")
        self.logger.info(f"最佳 mAP@0.5: {self.best_mAP:.4f}")
        self.logger.info(f"训练历史模型权重:")
        for i, weights in enumerate(self.model_weights_history):
            self.logger.info(f"  第{i}轮: {weights}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='YOLOv10半监督迭代训练')
    parser.add_argument('--config', type=str, default='semi/config/train_config.yaml',
                        help='训练配置文件路径')
    args = parser.parse_args()

    # 创建训练器并开始训练
    trainer = IterativeTrainer(args.config)
    trainer.train()


if __name__ == '__main__':
    main()