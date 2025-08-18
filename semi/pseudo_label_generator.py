"""
伪标签生成器模块
用于使用训练好的YOLOv10模型对无标签数据生成伪标签
"""

import os
import cv2
import torch
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging


class PseudoLabelGenerator:
    """伪标签生成器类"""

    def __init__(self, config_from_trainer):
        """
        初始化伪标签生成器

        Args:
            config: 伪标签生成配置
        """
        self.config = config_from_trainer
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_model(self, weights_path):
        """
        加载YOLOv10模型

        Args:
            weights_path: 模型权重文件路径

        Returns:
            加载的模型
        """
        try:
            # 这里假设使用Ultralytics的YOLOv10
            from ultralytics import YOLOv10
            model = YOLOv10(weights_path)
            model.to(self.device)
            self.logger.info(f"成功加载模型: {weights_path}")
            return model
        except Exception as e:
            self.logger.error(f"加载模型失败: {e}")
            raise

    def convert_to_yolo_format(self, boxes, img_width, img_height):
        """
        将检测框转换为YOLO格式

        Args:
            boxes: 检测框列表 [(x1, y1, x2, y2, conf, class_id), ...]
            img_width: 图像宽度
            img_height: 图像高度

        Returns:
            YOLO格式的标签列表
        """
        yolo_labels = []

        for box in boxes:
            x1, y1, x2, y2, conf, class_id = box

            # 计算中心点坐标和宽高
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1

            # 归一化到[0, 1]
            x_center_norm = x_center / img_width
            y_center_norm = y_center / img_height
            width_norm = width / img_width
            height_norm = height / img_height

            # 确保值在合理范围内
            x_center_norm = np.clip(x_center_norm, 0, 1)
            y_center_norm = np.clip(y_center_norm, 0, 1)
            width_norm = np.clip(width_norm, 0, 1)
            height_norm = np.clip(height_norm, 0, 1)

            # YOLO格式: class_index x_center y_center width height
            yolo_label = f"{int(class_id)} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"
            yolo_labels.append(yolo_label)

        return yolo_labels

    def process_single_image(self, model, image_path, conf_threshold):
        """
        处理单张图片并生成伪标签，不仅返回YOLO标签，还返回包含置信度的元数据。

        Args:
            model: YOLOv10模型
            image_path: 图片路径
            conf_threshold: 置信度阈值

        Returns:
            YOLO格式的标签列表
        """
        try:
            # 读取图片
            img = cv2.imread(str(image_path))
            if img is None:
                self.logger.warning(f"无法读取图片: {image_path}")
                return []

            img_height, img_width = img.shape[:2]
            results = model(image_path, conf=conf_threshold, verbose=False)

            # 提取检测结果
            boxes_for_yolo = []
            objects_metadata = []
            for r in results:
                if hasattr(r, 'boxes') and r.boxes is not None:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls = int(box.cls[0].cpu().numpy())

                        if conf >= conf_threshold:
                            boxes_for_yolo.append((x1, y1, x2, y2, conf, cls))
                            pixel_w = x2 - x1
                            pixel_h = y2 - y1
                            objects_metadata.append({
                                "class_id": cls,
                                "confidence": round(conf, 4),
                                "bbox_pixels": [round(val) for val in [x1, y1, x2, y2]],
                                "area": int(pixel_w * pixel_h)
                            })

            yolo_labels = self.convert_to_yolo_format(boxes_for_yolo, img_width, img_height)
            return yolo_labels, objects_metadata

        except Exception as e:
            self.logger.error(f"处理图片 {image_path} 时出错: {e}")
            return [], []

    def generate(self, model, config):
        """
        生成伪标签的主函数

        Args:
            config: 包含以下键的配置字典
                - weights: 模型权重路径
                - source_images: 无标签图片目录
                - output_dir: 伪标签输出目录
                - conf_threshold: 置信度阈值
                - img_size: 推理图像尺寸

        Returns:
            生成的伪标签文件数量
        """
        # 提取配置
        conf_threshold = config['conf_threshold']
        source_images_dir = Path(config['source_images'])
        output_dir = Path(config['output_dir'])
        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)

        # 优先直接使用传入的模型对象，否则从路径加载
        self.logger.info("直接使用传入的模型对象进行伪标签生成。")
        model.conf = conf_threshold  # 为模型设置置信度阈值

        # 获取所有图片文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(source_images_dir.glob(f'*{ext}'))
            image_files.extend(source_images_dir.glob(f'*{ext.upper()}'))

        image_files = sorted(list(set(image_files)))  # 去重

        self.logger.info(f"找到 {len(image_files)} 张图片待处理")

        # 处理每张图片
        generated_count = 0

        for image_path in tqdm(image_files, desc="生成伪标签"):
            # 生成伪标签
            yolo_labels = self.process_single_image(model, image_path, conf_threshold)

            # 保存标签文件
            label_filename = image_path.stem + '.txt'
            label_path = output_dir / label_filename

            # 即使没有检测到目标，也创建空文件
            with open(label_path, 'w') as f:
                if yolo_labels:
                    f.write('\n'.join(yolo_labels))
                    generated_count += 1
                # 空文件表示该图片中没有目标

        self.logger.info(f"伪标签生成完成，共生成 {generated_count} 个包含目标的标签文件")

        return len(image_files)  # 返回处理的图片总数

    def generate_with_flexmatch_filter(self, model, config, iteration, thresholds_path):
        """
        [核心方法] 这是您在半监督循环中应该调用的方法。
        它加载在线训练最终生成的动态阈值，并用它来筛选伪标签。
        """
        # 1. 加载配置和动态阈值
        # 这个阈值是“海选”阈值，可以设置得相对低一些，例如0.25
        conf_threshold = config['pseudo_label_conf_threshold']
        source_images_dir = Path(config['unlabeled_images_dir'])

        # 动态构建输出路径，严格遵循您的config设计
        output_root = Path(config['pseudo_labels_output_root'])
        output_dir = output_root / f'iter_{iteration}'
        labels_output_dir = output_dir / 'labels'
        meta_output_dir = output_dir / 'labels_meta'
        labels_output_dir.mkdir(parents=True, exist_ok=True)
        meta_output_dir.mkdir(parents=True, exist_ok=True)

        # 加载在线训练最终生成的阈值文件
        try:
            with open(thresholds_path, 'r') as f:
                dynamic_thresholds = json.load(f)
            self.logger.info(f"成功加载动态阈值文件: {thresholds_path}")
        except FileNotFoundError:
            self.logger.error(f"错误: 动态阈值文件不存在于 {thresholds_path}。请确保在线训练已正确生成此文件。")
            raise

        # 2. 遍历所有无标签图片
        image_files = sorted(list(source_images_dir.glob('*.*')))
        self.logger.info(f"找到 {len(image_files)} 张图片待处理")

        model.model.eval()
        self.logger.info("模型已设置为评估模式 (eval mode) 以生成伪标签。")

        generated_count = 0
        for image_path in tqdm(image_files, desc=f"生成并筛选第 {iteration} 轮伪标签"):

            # 3. 生成候选伪标签和元数据
            candidate_labels, candidate_metadata = self.process_single_image(model, image_path, conf_threshold)
            if not candidate_metadata:
                continue

            # 4. 使用动态阈值进行筛选
            reliable_labels_indices = [
                idx for idx, meta_obj in enumerate(candidate_metadata)
                if meta_obj['confidence'] >= dynamic_thresholds[meta_obj['class_id']]
            ]

            # 5. 只保存通过筛选的高质量伪标签
            if reliable_labels_indices:
                final_yolo_labels = [candidate_labels[i] for i in reliable_labels_indices]
                final_metadata = [candidate_metadata[i] for i in reliable_labels_indices]

                base_filename = image_path.stem
                label_path = labels_output_dir / f"{base_filename}.txt"
                meta_path = meta_output_dir / f"{base_filename}.json"

                with open(label_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(final_yolo_labels))
                with open(meta_path, 'w', encoding='utf-8') as f:
                    json.dump(final_metadata, f, indent=2)

                generated_count += 1

        self.logger.info(f"伪标签筛选和生成完成，共为 {generated_count} 张图片生成了高质量伪标签。")
        return str(output_dir)

    def batch_generate(self, config, batch_size=32):
        """
        批量生成伪标签（用于加速处理）

        Args:
            config: 生成配置
            batch_size: 批处理大小

        Returns:
            生成的伪标签文件数量
        """
        # 提取配置
        weights_path = config['weights']
        source_images_dir = Path(config['source_images'])
        output_dir = Path(config['output_dir'])
        conf_threshold = config['conf_threshold']
        img_size = config.get('img_size', 640)

        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)

        # 加载模型
        model = self.load_model(weights_path)

        # 获取所有图片文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(source_images_dir.glob(f'*{ext}'))
            image_files.extend(source_images_dir.glob(f'*{ext.upper()}'))

        self.logger.info(f"找到 {len(image_files)} 张图片待处理")

        # 批量处理
        generated_count = 0

        for i in tqdm(range(0, len(image_files), batch_size), desc="批量生成伪标签"):
            batch_files = image_files[i:i + batch_size]

            # 批量推理
            results = model(batch_files, conf=conf_threshold, imgsz=img_size, verbose=False)

            # 处理每个结果
            for j, (image_path, result) in enumerate(zip(batch_files, results)):

                # 获取图片尺寸
                img = cv2.imread(str(image_path))
                if img is None:
                    continue

                img_height, img_width = img.shape[:2]

                # 提取检测框
                boxes = []
                if result.boxes is not None:

                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = box.cls[0].cpu().numpy()

                        if conf >= conf_threshold:
                            boxes.append((x1, y1, x2, y2, conf, cls))


                # 转换为YOLO格式
                yolo_labels = self.convert_to_yolo_format(boxes, img_width, img_height)

                # 保存标签文件
                label_filename = image_path.stem + '.txt'
                label_path = output_dir / label_filename

                with open(label_path, 'w') as f:
                    if yolo_labels:
                        f.write('\n'.join(yolo_labels))
                        generated_count += 1

        self.logger.info(f"批量伪标签生成完成，共生成 {generated_count} 个包含目标的标签文件")

        return len(image_files)