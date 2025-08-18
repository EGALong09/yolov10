# custom_dataset.py
import json
import numpy as np
from pathlib import Path
import logging
import random
from ultralytics.data.dataset import YOLODataset
from semi.semi_augment import build_strong_transforms, build_weak_transforms
from ultralytics.data.augment import Compose, LetterBox


logger = logging.getLogger(__name__)


class FlexMatchInspiredDataset(YOLODataset):
    """
    一个实现了FlexMatch伪标签在线筛选的数据集类。
    """

    def __init__(self, *args, **kwargs):
        self.mode = kwargs.get('mode', 'train')
        # 步骤 1: 从 kwargs 中弹出我们自定义的参数，避免传递给父类。
        self.custom_args = {
            'use_strong_augment_for_retrain': kwargs.pop('use_strong_augment_for_retrain', True),
            'flexmatch_config': kwargs.pop('flexmatch_config', {}),
            'unlabeled_images_dir': kwargs.pop('unlabeled_images_dir', None),
            'generated_dataset_yamls_output_dir': kwargs.pop('generated_dataset_yamls_output_dir', None),
            'nc': kwargs.get('nc', 8)
        }

        # 步骤 2: 【核心修正】在调用 super().__init__() 之前，设置好 get_labels() 需要的所有属性。
        # 父类的 __init__ 会调用 get_labels()，所以这些属性必须提前存在。
        self.use_flexmatch = self.custom_args['unlabeled_images_dir'] is not None and self.mode == 'train'

        imgsz = kwargs.get('imgsz', 640)  # 从参数中预读 imgsz
        self.mosaic_border = [-imgsz // 2, -imgsz // 2]

        if self.use_flexmatch:
            flexmatch_cfg = self.custom_args['flexmatch_config']

            output_dir_path = self.custom_args['generated_dataset_yamls_output_dir']
            if output_dir_path is None:
                raise ValueError(
                    "配置错误: 在半监督模式下，'generated_dataset_yamls_output_dir' 必须被提供。"
                )

            generated_config_dir = Path(output_dir_path)
            initial_threshold = flexmatch_cfg.get('initial_threshold', 0.95)

            # 因为 self.data 此时还不存在，所以无法获取 nc。
            # 这部分逻辑主要是为了在训练开始前，如果阈值文件不存在，就创建一个。
            # 我们可以用一个临时的 nc 值来创建，后续会被回调函数更新。
            temp_nc_for_init = self.custom_args.get('nc')  # 假设从配置中传入或使用默认值

            self.thresholds_path = generated_config_dir / 'flexmatch_thresholds.json'
            if not self.thresholds_path.exists():
                self.dynamic_thresholds = [initial_threshold] * temp_nc_for_init
                generated_config_dir.mkdir(parents=True, exist_ok=True)
                with open(self.thresholds_path, 'w') as f:
                    json.dump(self.dynamic_thresholds, f)

            self.small_target_area_thresh = flexmatch_cfg.get('small_target_area_thresh', 32 * 32)

        # 步骤 3: 现在可以安全地调用父类的构造函数了。
        # 它会完成所有基础设置，并调用我们重写的 get_labels() 方法。
        super().__init__(*args, **kwargs)

        # 步骤 4: 如果是半监督模式，在 super() 调用后，self.data 已被赋值。
        # 此时，我们重新加载一次阈值文件，并可以校验其类别数是否正确。
        if self.use_flexmatch and self.thresholds_path.exists():
            with open(self.thresholds_path, 'r') as f:
                self.dynamic_thresholds = json.load(f)
            # 校验长度是否匹配
            if len(self.dynamic_thresholds) != self.data['nc']:
                logger.warning(
                    f"阈值文件中的类别数 ({len(self.dynamic_thresholds)}) 与数据集中的类别数 ({self.data['nc']}) 不匹配。请检查配置。")

    def get_labels(self):
        labels_list = super().get_labels()

        # 如果不是半监督模式（例如，在验证集或基准训练时），直接返回原始标签
        if not self.use_flexmatch:
            return labels_list

        if not hasattr(self, 'dynamic_thresholds'):
            with open(self.thresholds_path, 'r') as f:
                self.dynamic_thresholds = json.load(f)

        for i in range(len(labels_list)):
            labels_list[i]['use_weak_augment'] = False
            im_file = Path(labels_list[i]['im_file'])

            is_pseudo_labeled_sample = 'consolidated_data' in str(im_file)
            if not is_pseudo_labeled_sample:
                continue

            meta_path = Path(str(im_file).replace('/images/', '/labels_meta/')).with_suffix('.json')
            if not meta_path.exists() or meta_path.stat().st_size == 0:
                continue
            try:
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not metadata:
                continue

            reliable_indices = []
            original_cls = labels_list[i]['cls']
            for idx in range(len(original_cls)):
                if idx >= len(metadata):
                    break
                class_id = int(original_cls[idx][0])
                confidence = metadata[idx]['confidence']
                if class_id < len(self.dynamic_thresholds) and confidence >= self.dynamic_thresholds[class_id]:
                    reliable_indices.append(idx)

            reliable_metadata = [metadata[j] for j in reliable_indices]
            for meta_obj in reliable_metadata:
                if meta_obj.get('area', float('inf')) < self.small_target_area_thresh:
                    labels_list[i]['use_weak_augment'] = True
                    break

            labels_list[i]['cls'] = labels_list[i]['cls'][reliable_indices]
            labels_list[i]['bboxes'] = labels_list[i]['bboxes'][reliable_indices]
            if len(labels_list[i].get('segments', [])) > 0:
                labels_list[i]['segments'] = [labels_list[i]['segments'][j] for j in reliable_indices]

        return labels_list

    def build_transforms(self, hyp=None):

        if self.mode == 'val':
            logger.info("[增强策略] 已进入【验证模式】，使用标准验证集增强。")
            transforms = Compose([LetterBox(self.imgsz, auto=False, stride=self.stride)])
            transforms.append(self.get_formatter(hyp=hyp))
            return transforms

        use_dynamic_switching = self.custom_args.get('use_strong_augment_for_retrain', True)

        if self.augment:
            if use_dynamic_switching:
                # 半监督再训练: 启用动态决策
                logger.info("[增强策略] 已进入【半监督动态增强】模式。")
                return DynamicAugmentation(self, self.imgsz, hyp)
            else:
                # 基准训练: 直接使用标准的强增强
                logger.info("[增强策略] 已进入【基准模型标准增强】模式 (不含动态切换)。")
                transforms = build_strong_transforms(self, self.imgsz, hyp)
                transforms.append(self.get_formatter(hyp=hyp))
                return transforms
        else:
            # 总开关关闭: 使用最基础的弱增强
            logger.info("[增强策略] augment=False，使用【弱增强】模式。")
            transforms = build_weak_transforms(self, self.imgsz, hyp)
            transforms.append(self.get_formatter(hyp=hyp))
            return transforms



    def get_formatter(self, hyp=None):
        """辅助函数，用于获取标准的格式化转换器。"""
        from ultralytics.data.augment import Format
        return Format(
            bbox_format="xywh",
            normalize=True,
            return_mask=self.use_segments,
            return_keypoint=self.use_keypoints,
            return_obb=self.use_obb,
            batch_idx=True,
            mask_ratio=getattr(hyp, 'mask_ratio', 4),
            mask_overlap=getattr(hyp, 'overlap_mask', True),
        )


# 一个特殊的、可调用的类，用于封装动态决策逻辑
class DynamicAugmentation:
    def __init__(self, dataset, imgsz, hyp):
        self.strong_pipeline = build_strong_transforms(dataset, imgsz, hyp)
        self.weak_pipeline = build_weak_transforms(dataset, imgsz, hyp)
        self.formatter = dataset.get_formatter(hyp)
        self.use_strong_aug_config = dataset.custom_args.get('use_strong_augment_for_retrain', True)

    def __call__(self, labels):
        # --- 日志打印 ---
        # 同样使用采样的方式避免刷屏
        if random.random() < 0.01:  # 1%的概率打印
            image_name = Path(labels['im_file']).name
            # 根据最终决策来打印日志
            if self.use_strong_aug_config and not labels.get('use_weak_augment', False):
                logger.info(f"[动态增强决策] 图片: {image_name} -> 策略:【强增强】")
            else:
                reason = "包含小目标" if labels.get('use_weak_augment', False) else "总开关关闭"
                logger.info(f"[动态增强决策] 图片: {image_name} -> 策略:【弱增强】 (原因: {reason})")
        # ------------------------------------

        # 实际的决策执行逻辑
        if self.use_strong_aug_config and not labels.get('use_weak_augment', False):
            augmented_data = self.strong_pipeline(labels)
        else:
            augmented_data = self.weak_pipeline(labels)

        # 最终统一进行格式化
        return self.formatter(augmented_data)