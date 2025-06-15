"""
数据整合器模块
用于将原始有标签数据与伪标签数据合并，创建新的数据集
"""

import os
import shutil
import yaml
from pathlib import Path
import logging
from tqdm import tqdm


class DataConsolidator:
    """数据整合器类"""

    def __init__(self, config_from_trainer):
        """
        初始化数据整合器

        Args:
            config: 数据配置
        """
        self.config = config_from_trainer
        self.logger = logging.getLogger(__name__)

    def copy_files(self, source_dir, dest_dir, file_pattern="*"):
        """
        复制文件从源目录到目标目录

        Args:
            source_dir: 源目录
            dest_dir: 目标目录
            file_pattern: 文件匹配模式

        Returns:
            复制的文件数量
        """
        source_dir = Path(source_dir)
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        files = list(source_dir.glob(file_pattern))
        copied_count = 0

        for file_path in tqdm(files, desc=f"复制文件从 {source_dir.name}"):
            try:
                dest_path = dest_dir / file_path.name
                shutil.copy2(file_path, dest_path)
                copied_count += 1
            except Exception as e:
                self.logger.error(f"复制文件 {file_path} 失败: {e}")

        return copied_count

    def match_images_and_labels(self, images_dir, labels_dir):
        """
        匹配图片和标签文件

        Args:
            images_dir: 图片目录
            labels_dir: 标签目录

        Returns:
            匹配的文件对列表 [(image_path, label_path), ...]
        """
        images_dir = Path(images_dir)
        labels_dir = Path(labels_dir)

        # 获取所有标签文件
        label_files = list(labels_dir.glob("*.txt"))
        matched_pairs = []

        # 支持的图片格式
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

        for label_path in label_files:
            label_stem = label_path.stem

            # 查找对应的图片文件
            image_found = False
            for ext in image_extensions:
                # 尝试小写和大写扩展名
                for extension in [ext, ext.upper()]:
                    image_path = images_dir / f"{label_stem}{extension}"
                    if image_path.exists():
                        matched_pairs.append((image_path, label_path))
                        image_found = True
                        break
                if image_found:
                    break

            if not image_found:
                self.logger.warning(f"未找到标签 {label_path} 对应的图片")

        return matched_pairs

    def consolidate(self, consolidate_run_config):
        """
        整合数据的主函数

        Args:
            config: 包含以下键的配置字典
                - original_labeled_dir: 原始有标签数据目录
                - unlabeled_images_dir: 无标签图片目录
                - pseudo_labels_dir: 伪标签目录
                - output_dir: 输出目录
                - output_yaml: 输出的数据集配置文件路径
                - validation_path: 验证集路径

        Returns:
            生成的数据集配置文件路径
        """
        # 提取已解析好的绝对路径
        original_labeled_data_root_dir = Path(consolidate_run_config['original_labeled_data_root_dir'])
        unlabeled_images_input_dir = Path(consolidate_run_config['unlabeled_images_input_dir'])
        pseudo_labels_input_dir = Path(consolidate_run_config['pseudo_labels_input_dir'])

        consolidated_output_dir = Path(consolidate_run_config['consolidated_output_dir'])  # 合并数据的根目录
        new_dataset_yaml_path = Path(consolidate_run_config['new_dataset_yaml_path'])  # 新YAML的保存位置

        fixed_validation_images_path = Path(consolidate_run_config['fixed_validation_images_path'])
        fixed_validation_labels_path_str = consolidate_run_config.get('fixed_validation_labels_path')  # 可能为None
        original_dataset_yaml_for_names = consolidate_run_config['original_dataset_yaml_for_names']

        # 创建输出目录结构
        images_train_dir = consolidated_output_dir / 'images' / 'train'
        labels_train_dir = consolidated_output_dir / 'labels' / 'train'

        # 在创建前，如果目录已存在，先清空，避免旧的符号链接干扰
        if consolidated_output_dir.exists():
            shutil.rmtree(consolidated_output_dir)

        images_train_dir.mkdir(parents=True, exist_ok=True)
        labels_train_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("开始整合数据...")

        # 1. 创建指向原始有标签数据的符号链接
        self.logger.info("创建原始有标签数据的符号链接...")

        original_images_dir = original_labeled_data_root_dir / 'images' / 'train'
        original_labels_dir = original_labeled_data_root_dir / 'labels' / 'train'

        linked_original_images = 0
        for src_img_path in tqdm(original_images_dir.glob('*.*'), desc="链接原始图片"):
            dst_link_path = images_train_dir / src_img_path.name
            try:
                os.symlink(src_img_path.resolve(), dst_link_path)
                linked_original_images += 1
            except Exception as e:
                self.logger.error(f"创建符号链接失败: {src_img_path} -> {dst_link_path}. Error: {e}")

        linked_original_labels = 0
        for src_lbl_path in tqdm(original_labels_dir.glob('*.txt'), desc="链接原始标签"):
            dst_link_path = labels_train_dir / src_lbl_path.name
            try:
                os.symlink(src_lbl_path.resolve(), dst_link_path)
                linked_original_labels += 1
            except Exception as e:
                self.logger.error(f"创建符号链接失败: {src_lbl_path} -> {dst_link_path}. Error: {e}")

        self.logger.info(f"链接了 {linked_original_images} 张原始图片和 {linked_original_labels} 个原始标签")

        # 2. 链接无标签图片并复制伪标签
        self.logger.info("链接无标签图片并复制伪标签...")

        pseudo_pairs = self.match_images_and_labels(
            images_dir=unlabeled_images_input_dir,
            labels_dir=pseudo_labels_input_dir
        )

        added_pseudo_images = 0
        added_pseudo_labels = 0
        for image_path, label_path in tqdm(pseudo_pairs, desc="处理伪标签数据"):
            if label_path.stat().st_size > 0:
                try:
                    # 为图片创建符号链接
                    dst_img_link_path = images_train_dir / image_path.name
                    os.symlink(image_path.resolve(), dst_img_link_path)
                    added_pseudo_images += 1

                    # 复制伪标签文件
                    shutil.copy2(label_path, labels_train_dir / label_path.name)
                    added_pseudo_labels += 1
                except Exception as e:
                    self.logger.error(f"处理伪标签样本 {image_path.name} 时失败: {e}")

        self.logger.info(f"新增了 {added_pseudo_images} 张带有伪标签的图片和 {added_pseudo_labels} 个伪标签文件")

        # 3. 创建新的数据集配置文件 (new_dataset_yaml_path)
        self.logger.info("创建新的数据集配置文件...")

        # 读取原始配置文件以获取类别信息
        # 获取类别名
        with open(original_dataset_yaml_for_names, 'r', encoding='utf-8') as f:
            names_config = yaml.safe_load(f)
            class_names = names_config.get('names', {})
            if not class_names:
                # 默认类别以防万一
                class_names = {0: 'object'}
                self.logger.warning(f"从 {original_dataset_yaml_for_names} 未获取到类别名，使用默认值。")

        # --- 关键路径处理 ---
        # new_dataset_yaml_path 是新YAML的绝对路径
        # consolidated_output_dir 是合并后数据集的根目录 (绝对路径)
        # fixed_validation_images_path 是固定验证集图片文件夹的绝对路径
        # 'path' 字段：应该是 consolidated_output_dir 相对于 new_dataset_yaml_path 所在目录的相对路径
        yaml_file_parent_dir = new_dataset_yaml_path.parent
        path_field_for_new_yaml = os.path.relpath(consolidated_output_dir, yaml_file_parent_dir)
        # 'val' 字段：应该是 fixed_validation_images_path 相对于 consolidated_output_dir 的相对路径
        # 因为YOLO加载时是 yaml_dir/path_field/val_field
        val_field_for_new_yaml = os.path.relpath(fixed_validation_images_path, consolidated_output_dir)
        consolidated_output_dir_abs = Path(consolidate_run_config['consolidated_output_dir']).resolve()
        new_dataset_yaml_path_abs = Path(consolidate_run_config['new_dataset_yaml_path']).resolve()
        fixed_validation_images_path_abs = Path(consolidate_run_config['fixed_validation_images_path']).resolve()
        original_dataset_yaml_for_names = consolidate_run_config['original_dataset_yaml_for_names']

        # 创建新的配置
        dataset_config_for_new_yaml = {
            'path': str(consolidated_output_dir_abs), # 训练和（未来可能的）测试图片/标签的根目录
            'train': 'images/train',                  # 相对于上面的 'path'
            'val': str(fixed_validation_images_path_abs), # 验证集图片的绝对路径
            'names': class_names,
            'nc': len(class_names)  # 通常也需要类别数量
            # Ultralytics 通常会自动在 val 路径的同级寻找 labels 文件夹。
            # 如果你的验证集标签不在 fixed_validation_images_path 的 ../labels/val 这样的位置，
            # 你可能需要在 fixed_validation_images_path 直接指向包含 images 和 labels 的验证集根目录，
            # 然后在 initial_dataset_yaml 中正确配置。
            # 或者，如果YOLO允许，有时可以不写 path，直接给 train 和 val 写绝对路径。
            # 例如：
            # train: D:/path/to/consolidated_data/iter_N/images/train
            # val: D:/path/to/original_labeled_data/images/val
        }

        # 如果也提供了验证集标签路径 (通常YOLO会自动找，但可以明确)
        # if fixed_validation_labels_path_str:
        #     val_labels_field = os.path.relpath(Path(fixed_validation_labels_path_str), consolidated_output_dir)
        #     dataset_config_for_new_yaml['val_labels'] = str(Path(val_labels_field)) # 自定义字段，YOLO可能不认

        # 保存配置文件
        with open(new_dataset_yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(dataset_config_for_new_yaml, f, default_flow_style=False, allow_unicode=True)

        self.logger.info(f"新的数据集配置文件已保存到: {new_dataset_yaml_path}")

        # 4. 统计信息
        total_images = len(list(images_train_dir.glob('*.*')))
        total_labels = len(list(labels_train_dir.glob('*.txt')))

        self.logger.info(f"数据整合完成统计:")
        self.logger.info(f"  - 总图片数: {total_images}")
        self.logger.info(f"  - 总标签数: {total_labels}")

        return str(new_dataset_yaml_path)

    def validate_dataset(self, dataset_yaml_path):
        """
        验证数据集的完整性

        Args:
            dataset_yaml_path: 数据集配置文件路径

        Returns:
            验证结果字典
        """
        with open(dataset_yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        dataset_root = Path(dataset_yaml_path).parent / config['path']
        train_images_dir = dataset_root / config['train']
        train_labels_dir = train_images_dir.parent.parent / 'labels' / 'train'

        # 统计文件数量
        image_files = list(train_images_dir.glob('*.*'))
        label_files = list(train_labels_dir.glob('*.txt'))

        # 检查匹配情况
        image_stems = {f.stem for f in image_files}
        label_stems = {f.stem for f in label_files}

        # 找出不匹配的文件
        images_without_labels = image_stems - label_stems
        labels_without_images = label_stems - image_stems

        validation_result = {
            'total_images': len(image_files),
            'total_labels': len(label_files),
            'matched_pairs': len(image_stems & label_stems),
            'images_without_labels': len(images_without_labels),
            'labels_without_images': len(labels_without_images),
            'is_valid': len(images_without_labels) == 0 and len(labels_without_images) == 0
        }

        if not validation_result['is_valid']:
            self.logger.warning(f"数据集验证发现问题:")
            self.logger.warning(f"  - {validation_result['images_without_labels']} 张图片没有对应标签")
            self.logger.warning(f"  - {validation_result['labels_without_images']} 个标签没有对应图片")

        return validation_result

    def create_train_val_split(self, dataset_dir, val_ratio=0.2, random_seed=42):
        """
        从训练集中分割出验证集

        Args:
            dataset_dir: 数据集目录
            val_ratio: 验证集比例
            random_seed: 随机种子
        """
        import random
        random.seed(random_seed)

        dataset_dir = Path(dataset_dir)
        train_images_dir = dataset_dir / 'images' / 'train'
        train_labels_dir = dataset_dir / 'labels' / 'train'
        val_images_dir = dataset_dir / 'images' / 'val'
        val_labels_dir = dataset_dir / 'labels' / 'val'

        # 创建验证集目录
        val_images_dir.mkdir(parents=True, exist_ok=True)
        val_labels_dir.mkdir(parents=True, exist_ok=True)

        # 获取所有训练图片
        image_files = list(train_images_dir.glob('*.*'))
        random.shuffle(image_files)

        # 计算验证集大小
        val_size = int(len(image_files) * val_ratio)
        val_files = image_files[:val_size]

        # 移动文件到验证集
        moved_count = 0
        for image_path in tqdm(val_files, desc="创建验证集"):
            # 移动图片
            dest_image = val_images_dir / image_path.name
            shutil.move(str(image_path), str(dest_image))

            # 移动对应的标签
            label_path = train_labels_dir / f"{image_path.stem}.txt"
            if label_path.exists():
                dest_label = val_labels_dir / label_path.name
                shutil.move(str(label_path), str(dest_label))
                moved_count += 1

        self.logger.info(f"从训练集中分割出 {moved_count} 个样本作为验证集")

        return moved_count