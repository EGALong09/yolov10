
"""
基于YOLOv10的半监督迭代训练主控制脚本
用于管理整个迭代训练流程，包括基准模型训练、伪标签生成、数据整合和模型再训练
"""

import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import sys
import yaml
import argparse
import logging
from pathlib import Path
from datetime import datetime
from copy import deepcopy
from ultralytics import YOLOv10

# 导入工具模块
from semi.pseudo_label_generator import PseudoLabelGenerator
from semi.data_consolidator import DataConsolidator
from semi.model_evaluator import ModelEvaluator
from semi.config_manager import ConfigManager
from semi.callbacks import update_thresholds_standalone
from semi.model_trainer import ModelTrainer


class IterativeTrainer:
    """半监督迭代训练控制器"""

    def __init__(self, config_path):
        """
        初始化迭代训练器

        Args:
            config_path: 配置文件路径
        """
        self.raw_config = ConfigManager.load_config(config_path)
        self.config = self._resolve_paths(self.raw_config)
        self.setup_logging()

        try:
            # self.config['initial_dataset_yaml'] 已经是被 _resolve_paths 处理过的绝对路径
            with open(self.config['initial_dataset_yaml'], 'r', encoding='utf-8') as f:
                data_yaml = yaml.safe_load(f)
                num_classes = int(data_yaml['nc'])
                self.config['num_classes'] = num_classes  # 将 nc 存入 config 字典
                self.logger.info(f"从 {self.config['initial_dataset_yaml']} 成功读取到类别数量: {num_classes}")
        except Exception as e:
            self.logger.error(f"无法从 {self.config['initial_dataset_yaml']} 读取类别数量 'nc'，请检查文件。错误: {e}")
            raise

        self.trainer = ModelTrainer(self.config.get('yolo_config', {}))
        self.pseudo_generator = PseudoLabelGenerator(self.config)
        self.data_consolidator = DataConsolidator(self.config)
        self.evaluator = ModelEvaluator(self.config.get('eval_config', {}))

        self.student_model = None
        self.teacher_model = None
        self.current_iteration = 0
        self.best_mAP = 0.0
        self.previous_mAP = 0.0
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

        try:
            with open(self.config['initial_dataset_yaml_str'], 'r') as f:
                data_yaml = yaml.safe_load(f)
                num_classes = int(data_yaml['nc'])
                self.logger.info(f"从 {self.config['initial_dataset_yaml']} 中读取到类别数量: {num_classes}")
        except Exception as e:
            self.logger.error(f"无法从 {self.config['initial_dataset_yaml']} 读取类别数量'nc'，错误: {e}")
            raise

        # 从主配置的 yolo_config 块中获取 nwdloss 和 iou_ratio
        nwd_enabled = self.config.get('yolo_config', {}).get('nwdloss', False)
        iou_r = self.config.get('yolo_config', {}).get('iou_ratio', 0.5)

        # 配置基准模型训练参数
        baseline_config = {
            'data': self.config['initial_dataset_yaml_str'],
            'epochs': self.config['baseline_epochs'],
            'batch': self.config['batch_size'],
            'imgsz': self.config['image_size'],
            'project': str(Path(self.config['project_root']) / 'runs' / 'train_baseline'),
            'name': 'exp',
            'exist_ok': True,
            'nwdloss': nwd_enabled,
            'iou_ratio': iou_r,
            'nc': num_classes
        }

        # 直接将 self.student_model 对象传入训练，训练器不会启用EMA
        self.student_model.custom_is_baseline_run = True
        model_path = self.trainer.train(self.student_model, baseline_config)

        self.teacher_model = YOLOv10(model_path)

        # 别忘了将新创建的教师模型也移动到正确的设备上
        train_device = self.config.get('yolo_config', {}).get('device', 'cpu')
        self.teacher_model.to(train_device)

        self.model_weights_history.append(model_path)
        metrics = self.evaluator.evaluate(model_path, self.config['initial_dataset_yaml_str'])

        self.logger.info(f"基准模型训练完毕，教师模型已与学生模型同步。最佳权重: {model_path}")  # <-- 这是我添加的，便于您确认
        self.logger.info(f"基准模型评估完成，mAP@0.5: {metrics['mAP50']:.4f}")
        self.best_mAP = metrics['mAP50']
        return model_path

    def generate_pseudo_labels_with_filter(self, iteration, thresholds_path):
        """
        阶段2: 教师模型生成伪标签

        Args:
            model_path: 模型权重路径
            iteration: 当前迭代轮数
        """
        self.logger.info(f"开始生成第{iteration}轮伪标签")

        # 在生成之前，清理或创建干净的目标目录
        pseudo_labels_iter_root = Path(self.config['pseudo_labels_output_root']) / f'iter_{iteration}'

        if pseudo_labels_iter_root.exists():
            self.logger.warning(f"发现已存在的伪标签目录 {pseudo_labels_iter_root}，将进行清理...")
            shutil.rmtree(pseudo_labels_iter_root)

        # Generator内部会再次创建子目录，这里确保根目录干净即可
        pseudo_labels_iter_root.mkdir(parents=True)

        # 设置伪标签输出目录
        # 使用配置中 pseudo_labels_output_root
        # pseudo_labels_iter_root = Path(self.config['pseudo_labels_output_root']) / f'iter_{iteration}'
        # pseudo_labels_dir_path = pseudo_labels_iter_root / 'labels'  # 伪标签存在labels子文件夹
        # pseudo_labels_dir_path.mkdir(parents=True, exist_ok=True)
        #
        # # 生成伪标签
        # pseudo_label_config = {
        #     'source_images': self.config['unlabeled_images_dir_str'],
        #     'output_dir': str(pseudo_labels_dir_path),
        #     'conf_threshold': self.config['pseudo_label_conf_threshold'],
        #     'img_size': self.config['image_size']
        # }

        # 【关键】直接将教师模型对象传入
        pseudo_labeling_model = deepcopy(self.teacher_model)
        # num_generated = self.pseudo_generator.generate(pseudo_labeling_model, pseudo_label_config)

        pseudo_labels_dir_str = self.pseudo_generator.generate_with_flexmatch_filter(
            model=pseudo_labeling_model,
            config=self.config,  # 传递完整的config
            iteration=iteration,
            thresholds_path=thresholds_path
        )

        return pseudo_labels_dir_str

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

    def retrain_model(self, current_model_path, dataset_yaml, iteration):
        """
        阶段4: 模型再训练
        使用合并后的数据集训练新的YOLOv10模型

        Args:
            dataset_yaml: 数据集配置文件路径
            previous_model_path: 上一轮模型权重路径
            iteration: 当前迭代轮数
        """
        self.logger.info(f"开始第{iteration}轮模型再训练")

        # 从主配置的 yolo_config 块中获取 nwdloss 和 iou_ratio
        nwd_enabled = self.config.get('yolo_config', {}).get('nwdloss', False)
        iou_r = self.config.get('yolo_config', {}).get('iou_ratio', 0.5)

        project_for_yolo_runs = Path(self.config['project_root']) / 'runs'  # 训练输出仍在项目内
        # 确保学生模型从上一轮的最佳状态开始
        self.student_model.load(current_model_path)

        # 数据强增强配置
        use_strong_aug = self.config.get('use_strong_augment_for_retrain', False)
        if use_strong_aug:
            self.logger.info("再训练阶段已启用强数据增强 (Strong Augmentation)。")

        # 配置再训练参数
        retrain_config = {
            # ---- 基础训练参数 ----
            'data': dataset_yaml,
            'epochs': self.config['retrain_epochs'],
            'batch_size': self.config['batch_size'],
            'imgsz': self.config['image_size'],
            'project': str(project_for_yolo_runs / f'train_iter_{iteration}'),
            'name': 'exp',
            'exist_ok': True,
            'lr0': self.config.get('retrain_learning_rate', 0.001),  # 使用配置的学习率
            'cos_lr': True,
            'nwdloss': nwd_enabled,
            'iou_ratio': iou_r,
            'use_strong_augment': use_strong_aug,
            'nc': self.config.get('num_classes'),

            # ---- EMA 相关参数 ----
            'ema_teacher_enabled': True,  # 明确启用EMA
            'ema_decay': self.config.get('ema_decay', 0.9996),  # 传递decay值

            # --- FlexMatch所需参数 ---
            'unlabeled_images_dir': self.config['unlabeled_images_dir'],
            'flexmatch_config': self.config['flexmatch_config'],
            'generated_dataset_yamls_output_dir': self.config['generated_dataset_yamls_output_dir']  # 回调函数也需要这个

        }

        # 在调用 train 之前，附加半监督再训练的配置
        self.student_model.custom_is_baseline_run = False
        self.student_model.custom_train_args = {
            'use_strong_augment_for_retrain': self.config.get('use_strong_augment_for_retrain', True),
            'flexmatch_config': self.config.get('flexmatch_config'),
            'unlabeled_images_dir': self.config.get('unlabeled_images_dir'),
            'generated_dataset_yamls_output_dir': self.config.get('generated_dataset_yamls_output_dir')
        }
        new_student_model_path = self.trainer.train(self.student_model, retrain_config, self.teacher_model)

        self.model_weights_history.append(new_student_model_path)
        return new_student_model_path


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

        # 1. 更新历史最佳mAP，用于最终报告最好的模型
        if current_mAP > self.best_mAP:
            self.logger.info(f"新纪录！历史最佳mAP从 {self.best_mAP:.4f} 提升至 {current_mAP:.4f}")
            self.best_mAP = current_mAP
        else:
            self.logger.info(f"当前mAP {current_mAP:.4f} 未超过历史最佳 {self.best_mAP:.4f}")

        # 2. 计算与【上一轮】的性能提升，并基于此决定是否早停
        improvement_vs_previous = current_mAP - self.previous_mAP
        self.logger.info(
            f"相比【上一轮】模型提升: {improvement_vs_previous:+.4f} (上一轮mAP: {self.previous_mAP:.4f})")

        # 3. 更新 self.previous_mAP，为下一次迭代做准备
        self.previous_mAP = current_mAP

        # 4. 新的早停逻辑
        if not self.config.get('early_stopping', True):
            self.logger.info("早停已禁用，将继续迭代。")
            return True

        # 只有当性能相比上一轮没有明显提升时，才触发早停
        if improvement_vs_previous < self.config['min_improvement_threshold']:
            self.logger.warning(f"性能相比上一轮提升不明显({improvement_vs_previous:+.4f})，触发早停机制。")
            return False
        else:
            self.logger.info("性能相比上一轮有明显提升，继续迭代。")
            return True

    def train(self):
        """
        主训练函数：执行完整的半监督迭代训练流程
        """
        use_ssl = self.config.get('use_semi_supervised_learning', True)
        train_device = self.config.get('yolo_config', {}).get('device', 'cpu')

        if use_ssl:
            self.logger.info("模式: 半监督学习。开始迭代训练流程。")
            self.logger.info(f"总迭代轮数: {self.config['num_iterations']}")

            # 阶段0: 训练基准模型
            if self.config.get('skip_baseline_training', False):
                baseline_model_path = self.config.get('baseline_model_path')
                if not baseline_model_path or not Path(baseline_model_path).exists():
                    raise FileNotFoundError(f"Provided 'baseline_model_path' is invalid: {baseline_model_path}")

                self.logger.info(f"跳过训练，直接加载基准模型: {baseline_model_path}")
                self.student_model = YOLOv10(baseline_model_path)
                self.teacher_model = deepcopy(self.student_model)

                self.teacher_model.to(train_device)

                self.model_weights_history.append(baseline_model_path)

                if not self.config.get('skip_baseline_eval', False):
                    metrics = self.evaluator.evaluate(baseline_model_path, self.config['initial_dataset_yaml_str'])
                    self.best_mAP = metrics.get('mAP50', 0.0)
                    self.previous_mAP = self.best_mAP
                    self.logger.info(f"提供的基准模型初始 mAP@0.5: {self.best_mAP:.4f}")
                else:
                    self.best_mAP = 0.0
                    self.previous_mAP = 0.0  # 如果跳过评估，则从0开始
            else:
                self.student_model = YOLOv10(self.config['pretrained_model'])
                baseline_model_path = self.train_baseline_model()
                self.previous_mAP = self.best_mAP

            # 迭代训练循环
            current_model_path = baseline_model_path

            for iteration in range(1, self.config['num_iterations'] + 1):
                self.current_iteration = iteration
                self.logger.info(f"\n{'=' * 50}")
                self.logger.info(f"开始第 {iteration}/{self.config['num_iterations']} 轮迭代")

                try:
                    # --- 阶段1.5: 更新动态阈值 ---
                    self.logger.info("--- 步骤 A: 更新动态阈值 ---")
                    # 使用上一轮的教师模型 (self.teacher_model) 来计算新阈值
                    thresholds_path = update_thresholds_standalone(
                        teacher_model=self.teacher_model,
                        config=self.config,
                        iteration=iteration
                    )
                    if not thresholds_path:
                        self.logger.warning(f"第 {iteration} 轮阈值更新失败，跳过本次迭代。")
                        continue

                    # 阶段2: 生成伪标签 (不再传递模型路径，内部使用self.teacher_model)
                    pseudo_labels_root_dir = self.pseudo_generator.generate_with_flexmatch_filter(
                        model=self.teacher_model,
                        config=self.config,
                        iteration=iteration,
                        thresholds_path=thresholds_path  # 传入阈值文件路径
                    )

                    # 阶段3: 数据整合
                    dataset_yaml = self.consolidate_data(pseudo_labels_root_dir, iteration)

                    # 阶段4: 模型再训练 (传入的是上一轮学生模型的路径)
                    new_student_model_path = self.retrain_model(current_model_path, dataset_yaml, iteration)

                    # 阶段5: 评估并决定是否继续 (评估的是新的学生模型)
                    should_continue = self.evaluate_and_decide(new_student_model_path, iteration)

                    # 更新当前模型路径以进行下一轮训练
                    current_model_path = new_student_model_path

                    # 如果性能不再提升，提前停止
                    if not should_continue and self.config.get('early_stopping', True) and iteration > 1:
                        self.logger.info(f"在第{iteration}轮性能未提升，触发早停机制")
                        break

                except Exception as e:

                    self.logger.error(f"第{iteration}轮迭代出错: {e}", exc_info=True)  # exc_info=True 打印更详细的traceback
                    if not self.config.get('continue_on_error', False):
                        raise

            # 训练完成，输出总结
            self.logger.info("\n" + "=" * 50)
            self.logger.info("半监督迭代训练完成！")
            self.logger.info(f"最佳 mAP@0.5: {self.best_mAP:.4f}")
            self.logger.info(f"训练历史模型权重:")
            for i, weights in enumerate(self.model_weights_history):
                self.logger.info(f"  第{i}轮: {weights}")

        else:
            self.logger.info("模式: 标准监督学习。将使用初始有标签数据集进行训练。")

            # 加载预训练模型
            model = YOLOv10(self.config['pretrained_model'])

            # 配置训练参数（直接使用基准模型的配置）
            supervised_config = {
                'data': self.config['initial_dataset_yaml_str'],
                'epochs': self.config['baseline_epochs'],
                'batch': self.config['batch_size'],
                'imgsz': self.config['image_size'],
                'project': str(Path(self.config['project_root']) / 'runs' / 'supervised_only'),
                'name': 'exp',
                'exist_ok': True
            }
            self.logger.info(f"训练参数: {supervised_config}")

            # 开始训练
            model_path = self.trainer.train(model, supervised_config)

            # 评估最终模型
            self.logger.info("标准监督训练完成。开始最终评估...")
            metrics = self.evaluator.evaluate(model_path, self.config['initial_dataset_yaml_str'])
            self.logger.info(f"最终模型评估完成，mAP@0.5: {metrics['mAP50']:.4f}")
            self.model_weights_history.append(model_path)

        self.logger.info("\n" + "=" * 50)
        self.logger.info("所有任务已完成！")
        self.logger.info(f"训练历史模型权重:")
        for i, weights in enumerate(self.model_weights_history):
            self.logger.info(f"  - {weights}")

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