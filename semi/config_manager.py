"""
配置管理器模块
用于管理和验证训练配置
"""

import yaml
import json
from pathlib import Path
import logging


class ConfigManager:
    """配置管理器类"""

    @staticmethod
    def load_config(config_path):
        """
        加载配置文件

        Args:
            config_path: 配置文件路径（支持yaml和json）

        Returns:
            配置字典
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        # 根据文件扩展名加载配置
        if config_path.suffix in ['.yaml', '.yml']:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        elif config_path.suffix == '.json':
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")

        # 验证配置
        ConfigManager.validate_config(config)

        # 设置默认值
        config = ConfigManager.set_defaults(config)

        return config

    @staticmethod
    def validate_config(config):
        """
        验证配置的必要字段

        Args:
            config: 配置字典
        """
        required_fields = [
            'project_root',
            'datasets_base_dir',
            'initial_dataset_yaml',
            # 'labeled_data_root_dir', # 可选，如果 initial_dataset_yaml 的 path 字段已足够
            'unlabeled_images_dir',
            'pseudo_labels_output_root',
            'consolidated_data_output_root',
            'generated_dataset_yamls_output_dir',
            'fixed_validation_images_path',  # 用于 DataConsolidator
            # fixed_validation_labels_path 是可选的，YOLO通常会自动找

            'pretrained_model',
            'baseline_epochs',
            'retrain_epochs',
            'batch_size',
            'image_size',
            'pseudo_label_conf_threshold',
            'num_iterations'
        ]

        missing_fields = []
        for field in required_fields:
            if field not in config:
                missing_fields.append(field)

        if missing_fields:
            raise ValueError(f"配置文件缺少必要字段: {missing_fields}")

    @staticmethod
    def set_defaults(config):
        """
        设置配置默认值

        Args:
            config: 配置字典

        Returns:
            包含默认值的配置字典
        """
        defaults = {
            # 训练参数默认值
            'retrain_learning_rate': 0.001,
            'min_improvement_threshold': 0.001,
            'early_stopping': True,
            'continue_on_error': True,

            # YOLO配置默认值
            'yolo_config': {
                'device': 0,
                'workers': 8,
                'amp': True,
                'patience': 50,
                'seed': 0,
                'optimizer': 'SGD',
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'warmup_epochs': 3.0,
                'close_mosaic': 10,
                'augment': True,
                'augment_params': {
                    'hsv_h': 0.015,
                    'hsv_s': 0.7,
                    'hsv_v': 0.4,
                    'degrees': 0.0,
                    'translate': 0.1,
                    'scale': 0.5,
                    'shear': 0.0,
                    'perspective': 0.0,
                    'flipud': 0.0,
                    'fliplr': 0.5,
                    'mosaic': 1.0,
                    'mixup': 0.0
                }
            },

            # 伪标签配置默认值
            'pseudo_label_config': {},

            # 数据配置默认值
            'data_config': {},

            # 评估配置默认值
            'eval_config': {
                'batch_size': 32,
                'imgsz': 640,
                'conf': 0.001,
                'iou': 0.6,
                'device': 0,
                'save_json': False,
                'save_txt': False
            }
        }

        # 对于新的路径配置，通常没有默认值，依赖用户在 train_config.yaml 中明确指定
        # 这里主要是确保如果用户省略了某些下一级的配置块（如 pseudo_label_config），它们会被创建为空字典
        config.setdefault('pseudo_label_config', {})
        config.setdefault('data_config', {})  # 这个可以用来传递 datasets_base_dir 给 DataConsolidator

        # 将 datasets_base_dir 传递给下一级配置，方便各模块使用
        config['data_config']['datasets_base_dir'] = config['datasets_base_dir']
        config['pseudo_label_config']['datasets_base_dir'] = config['datasets_base_dir']

        # 递归合并默认值
        return ConfigManager._merge_dicts(defaults, config)

    @staticmethod
    def _merge_dicts(default_dict, custom_dict):
        """
        递归合并两个字典

        Args:
            default_dict: 默认值字典
            custom_dict: 自定义值字典

        Returns:
            合并后的字典
        """
        result = default_dict.copy()

        for key, value in custom_dict.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigManager._merge_dicts(result[key], value)
            else:
                result[key] = value

        return result

    @staticmethod
    def save_config(config, save_path):
        """
        保存配置文件

        Args:
            config: 配置字典
            save_path: 保存路径
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if save_path.suffix in ['.yaml', '.yml']:
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        elif save_path.suffix == '.json':
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"不支持的配置文件格式: {save_path.suffix}")

    @staticmethod
    def create_default_config(save_path):
        """
        创建默认配置文件

        Args:
            save_path: 保存路径
        """
        default_config = {
            # 项目配置
            'project_root': './yolov10_ssl_project',

            # 数据集配置
            'initial_dataset_yaml': './data/dataset_labeled.yaml',
            'labeled_data_dir': './data/labeled',
            'unlabeled_images_dir': './data/unlabeled/images',
            'validation_data_path': './data/labeled/images/val',

            # 模型配置
            'pretrained_model': 'yolov10n.pt',  # 可选: yolov10n/s/m/l/x.pt

            # 训练配置
            'num_iterations': 3,
            'baseline_epochs': 50,
            'retrain_epochs': 30,
            'batch_size': 16,
            'image_size': 640,

            # 伪标签配置
            'pseudo_label_conf_threshold': 0.3,

            # 学习率配置
            'retrain_learning_rate': 0.001,

            # 早停配置
            'min_improvement_threshold': 0.001,
            'early_stopping': True,
            'continue_on_error': True,

            # YOLO训练配置
            'yolo_config': {
                'device': 0,  # GPU设备号，使用CPU则设为'cpu'
                'workers': 8,
                'amp': True,  # 自动混合精度
                'patience': 50,
                'seed': 0,
                'optimizer': 'SGD',
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'warmup_epochs': 3.0,
                'close_mosaic': 10,
                'augment': True,
                'augment_params': {
                    'hsv_h': 0.015,
                    'hsv_s': 0.7,
                    'hsv_v': 0.4,
                    'degrees': 0.0,
                    'translate': 0.1,
                    'scale': 0.5,
                    'shear': 0.0,
                    'perspective': 0.0,
                    'flipud': 0.0,
                    'fliplr': 0.5,
                    'mosaic': 1.0,
                    'mixup': 0.0
                }
            },

            # 评估配置
            'eval_config': {
                'batch_size': 32,
                'imgsz': 640,
                'conf': 0.001,
                'iou': 0.6,
                'device': 0,
                'save_json': False,
                'save_txt': False
            }
        }

        # 保存配置
        ConfigManager.save_config(default_config, save_path)

        logging.info(f"默认配置文件已创建: {save_path}")

        return default_config