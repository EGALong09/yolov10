"""
模型评估器模块
用于评估YOLOv10模型的性能
"""

import logging
from pathlib import Path
import json
import numpy as np
from ultralytics.utils import ops
from tqdm import tqdm
from ultralytics import YOLOv10

from ultralytics.models.yolo.detect import DetectionValidator # 导入要patch的类
from ultralytics.utils import LOGGER
# import types # 如果要用 MethodType 绑定，但直接赋值方法通常也可



class ModelEvaluator:
    """模型评估器类"""

    def __init__(self, config):
        """
        初始化模型评估器

        Args:
            config: 评估配置
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

    def evaluate(self, model_path, dataset_yaml):
        """
        评估模型性能

        Args:
            model_path: 模型权重路径
            dataset_yaml: 数据集配置文件路径

        Returns:
            评估指标字典
        """
        original_postprocess_backup = None  # 在try外部定义，确保finally中可访问

        try:
            # 加载模型
            model = YOLOv10(model_path)

            # 执行验证
            self.logger.info(f"开始评估模型: {model_path}")

            # --- Monkey Patching postprocess ---
            # 1. 保存原始的 postprocess 方法 (如果之前没保存过)
            #    使用一个独特的属性名来存储备份，避免冲突
            if not hasattr(DetectionValidator, '_original_postprocess_semi_train_backup'):
                DetectionValidator._original_postprocess_semi_train_backup = DetectionValidator.postprocess
                LOGGER.info("已备份原始的 DetectionValidator.postprocess 方法。")

            # original_postprocess_backup 变量用于在finally中恢复，确保它指向正确的原始方法
            original_postprocess_backup = DetectionValidator._original_postprocess_semi_train_backup

            # 2. 定义我们新的 postprocess 逻辑
            def patched_yolo10_postprocess(validator_self, preds_from_model_call):
                # validator_self 就是 DetectionValidator 实例 (即调用 postprocess 的那个 self)
                # preds_from_model_call 是 model() 的直接输出，即我们调试得到的字典

                preds_for_nms = None  # 初始化，用于NMS的张量

                if isinstance(preds_from_model_call, dict) and \
                    'one2one' in preds_from_model_call and \
                    isinstance(preds_from_model_call['one2one'], tuple) and \
                    len(preds_from_model_call['one2one']) > 0 and \
                    hasattr(preds_from_model_call['one2one'][0], 'shape'):

                    # 根据你的调试输出，'one2one'元组的第一个元素是 [4, 84, 9261] 的张量
                    # 这个张量的第二维 84 = 80 (classes) + 4 (xywh)
                    # 第三维 9261 是总的预测框数量
                    # 这个格式通常需要 permute 操作，NMS期望的是 [bs, num_boxes, num_classes + 4]
                    # 即 [4, 9261, 84]

                    raw_tensor = preds_from_model_call['one2one'][0]  # Shape: [4, 84, 9261]
                    LOGGER.info(f"Patched Postprocess: 使用 'one2one'[0] (形状: {raw_tensor.shape})")

                    # 进行维度转换 (permute) 以匹配NMS的期望输入格式
                    # NMS通常期望的格式是 (batch_size, num_predictions, num_classes + 4_coordinates)
                    if raw_tensor.ndim == 3 and raw_tensor.shape[1] == (validator_self.nc + 4):
                        preds_for_nms = raw_tensor.permute(0, 2, 1).contiguous()  # from [bs, nc+4, num_preds] to [bs, num_preds, nc+4]
                        LOGGER.info(f"Patched Postprocess: 已将 'one2one'[0] 维度转换为 {preds_for_nms.shape} 用于NMS。")
                    else:
                        LOGGER.error(f"Patched Postprocess: 'one2one'[0] 张量形状 {raw_tensor.shape} 不符合 [bs, nc+4, num_preds] 格式，无法正确转换。类别数 (nc)={validator_self.nc}")
                        return []  # 返回空结果或抛出错误，表示处理失败

                # 你也可以添加对 'one2many'[0] 的处理作为备选，如果 'one2one'[0] 格式不对或不存在
                # ... （省略了 one2many 的备选逻辑，可以后续添加） ...

                # 处理非YOLOv10字典输出的情况（例如，其他YOLO模型或标准格式）
                elif isinstance(preds_from_model_call, (list, tuple)):
                    if len(preds_from_model_call) > 0 and hasattr(preds_from_model_call[0], 'shape'):
                        preds_for_nms = preds_from_model_call[0]  # 通常取第一个元素
                        LOGGER.info(f"Patched Postprocess: 输入是列表/元组，使用第一个元素。形状:  {preds_for_nms.shape}")
                    else:
                        LOGGER.error("Patched Postprocess: 输入是列表/元组，但为空或第一个元素不是张量。")
                        return []
                elif hasattr(preds_from_model_call, 'shape'):  # 如果已经是单个张量
                    preds_for_nms = preds_from_model_call
                    LOGGER.info(f"Patched Postprocess: 输入已经是张量。形状: {preds_for_nms.shape}")
                else:
                    # 未知格式，记录错误并返回空列表
                    LOGGER.error(f"Patched Postprocess:输入类型 {type(preds_from_model_call)} 无法识别用于NMS处理。")
                    return []

                # 再次检查 preds_for_nms 是否有效
                if preds_for_nms is None or isinstance(preds_for_nms, dict):  # 再次检查，确保提取到了张量
                    LOGGER.error(f"Patched Postprocess: NMS的输入仍然有问题或是None: {type(preds_for_nms)}")
                    return []  # 返回空列表以避免NMS报错

                # 调用ops.non_max_suppression
                # 使用 validator_self 来访问 args, lb, nc 等属性
                return ops.non_max_suppression(
                    preds_for_nms,
                    validator_self.args.conf,
                    validator_self.args.iou,
                    labels=validator_self.lb,
                    multi_label=True,
                    agnostic=validator_self.args.agnostic_nms,
                    max_det=validator_self.args.max_det,
                    nc=validator_self.nc,
                    # nm=getattr(validator_self, 'nm', 0) # nm参数在较新版本中可能不存在或用法改变
                )

            # 3. 应用 patch，仅当当前模型是YOLOv10时（你可以通过model_path或其他方式判断）
            # 为了简单起见，我们先无条件patch，但理想情况下应有条件
            DetectionValidator.postprocess = patched_yolo10_postprocess
            LOGGER.info("已应用Monkey Patch到 DetectionValidator.postprocess 以进行YOLOv10评估。")

            # 验证参数
            val_args = {
                'data': dataset_yaml,
                'batch': self.config.get('batch_size', 32),
                'imgsz': self.config.get('imgsz', 640),
                'conf': self.config.get('conf', 0.001),
                'iou': self.config.get('iou', 0.6),
                'device': self.config.get('device', 0),
                'save_json': self.config.get('save_json', False),
                'save_txt': self.config.get('save_txt', False),
                'verbose': True
            }

            results = model.val(**val_args) # 现在调用 val()


        except Exception as e:
            self.logger.error(f"模型评估失败: {e}")
            # 即便发生异常，也尝试恢复原始方法
            if original_postprocess_backup is not None and hasattr(DetectionValidator, 'postprocess') and DetectionValidator.postprocess == patched_yolo10_postprocess:
                DetectionValidator.postprocess = original_postprocess_backup
                LOGGER.info("因评估过程中出错，已恢复原始的 DetectionValidator.postprocess 方法。")
            raise  # 重新抛出异常，以便主程序知道出错了
        finally:
            # 4. 恢复原始方法 (非常重要！)
            # 确保只在patch成功应用且原始方法被备份后才恢复
            if original_postprocess_backup is not None and hasattr(DetectionValidator,'postprocess') and DetectionValidator.postprocess == patched_yolo10_postprocess:
                DetectionValidator.postprocess = original_postprocess_backup
                # 清理备份属性，以便下次如果再次调用evaluate可以重新备份（尽管通常一个脚本实例只会patch一次）
                # if hasattr(DetectionValidator, '_original_postprocess_semi_train_backup'):
                #     del DetectionValidator._original_postprocess_semi_train_backup
                LOGGER.info("评估结束后，已恢复原始的 DetectionValidator.postprocess 方法。")

        # ... (提取指标的代码) ...
        # 确保这里能正确处理 results，即使验证过程中有部分失败（例如返回空metrics）
        metrics = {
            'precision': 0.0, 'recall': 0.0, 'mAP50': 0.0, 'mAP50-95': 0.0,
            'speed': {'preprocess': 0.0, 'inference': 0.0, 'postprocess': 0.0},
            'per_class': {}
        }
        if results and hasattr(results, 'box'):  # 检查 results 和 results.box 是否存在
            metrics['precision'] = float(results.box.mp) if hasattr(results.box, 'mp') else 0.0
            metrics['recall'] = float(results.box.mr) if hasattr(results.box, 'mr') else 0.0
            metrics['mAP50'] = float(results.box.map50) if hasattr(results.box, 'map50') else 0.0
            metrics['mAP50-95'] = float(results.box.map) if hasattr(results.box, 'map') else 0.0
            if hasattr(results, 'speed') and isinstance(results.speed, dict):
                metrics['speed']['preprocess'] = float(results.speed.get('preprocess', 0.0))
                metrics['speed']['inference'] = float(results.speed.get('inference', 0.0))
                metrics['speed']['postprocess'] = float(results.speed.get('postprocess', 0.0))

            if hasattr(results.box, 'ap_class_index') and hasattr(results.box, 'ap50') and hasattr(results, 'names'):
                class_metrics_data = {}
                ap_all_classes = getattr(results.box, 'ap', None)  # shape (num_classes, 10 iou thresholds) or similar
                for i, class_idx_tensor in enumerate(results.box.ap_class_index):
                    class_idx = int(class_idx_tensor.item())  # 转换为Python int
                    class_name = results.names[class_idx]
                    ap50_value = float(results.box.ap50[i].item())  # 转换为Python float

                    # 获取所有IoU阈值下的AP (mAP50-95 for this class)
                    # ap_all_classes 的第一维通常对应类别索引，第二维是不同iou阈值的AP值
                    # 我们需要该类别在所有IoU下的平均AP，这通常是 results.box.ap[class_idx] 的均值，
                    # 或者直接使用 results.box.per_class_ap (如果存在)
                    # 更简单的方式是，如果ap属性是每个类别在0.5:0.95下的map，直接使用
                    ap_value_for_class = 0.0
                    if ap_all_classes is not None and ap_all_classes.ndim >= 1 and i < ap_all_classes.shape[0]:
                        if ap_all_classes.ndim == 2 and ap_all_classes.shape[
                            1] > 0:  # [num_classes, num_iou_thresholds]
                            ap_value_for_class = float(ap_all_classes[i].mean().item()) if ap_all_classes[
                                                                                               i].numel() > 0 else 0.0
                        elif ap_all_classes.ndim == 1:  # [num_classes] - 假设这就是该类别的 mAP50-95
                            ap_value_for_class = float(ap_all_classes[i].item())

                    class_metrics_data[class_name] = {
                        'AP50': ap50_value,
                        'AP': ap_value_for_class  # 这是该类别在0.5:0.95 IoU范围内的平均AP
                    }
                metrics['per_class'] = class_metrics_data
        else:
            self.logger.warning("评估结果对象 'results' 或 'results.box' 无效，指标可能不完整。")

        self.logger.info(f"评估完成 - mAP@0.5: {metrics['mAP50']:.4f}, mAP@0.5:0.95: {metrics['mAP50-95']:.4f}")

        # 保存评估结果
        self._save_evaluation_results(model_path, metrics)

        return metrics

    def evaluate_with_cli(self, model_path, dataset_yaml):
        """
        使用命令行接口评估模型（备选方案）

        Args:
            model_path: 模型权重路径
            dataset_yaml: 数据集配置文件路径

        Returns:
            评估指标字典
        """
        import subprocess
        import yaml

        # 构建命令
        cmd = [
            'yolo', 'detect', 'val',
            f'model={model_path}',
            f'data={dataset_yaml}',
            f'batch={self.config.get("batch_size", 32)}',
            f'imgsz={self.config.get("imgsz", 640)}',
            f'device={self.config.get("device", 0)}'
        ]

        self.logger.info(f"执行评估命令: {' '.join(cmd)}")

        try:
            # 执行评估
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            # 解析输出获取指标
            output_lines = result.stdout.split('\n')
            metrics = self._parse_cli_output(output_lines)

            return metrics

        except subprocess.CalledProcessError as e:
            self.logger.error(f"评估命令执行失败: {e}")
            self.logger.error(f"错误输出: {e.stderr}")
            raise

    def compare_models(self, model_paths, dataset_yaml):
        """
        比较多个模型的性能

        Args:
            model_paths: 模型路径列表
            dataset_yaml: 数据集配置文件路径

        Returns:
            比较结果字典
        """
        comparison_results = {}

        for model_path in model_paths:
            model_name = Path(model_path).parent.parent.name
            self.logger.info(f"评估模型: {model_name}")

            try:
                metrics = self.evaluate(model_path, dataset_yaml)
                comparison_results[model_name] = metrics
            except Exception as e:
                self.logger.error(f"评估模型 {model_name} 失败: {e}")
                comparison_results[model_name] = None

        # 找出最佳模型
        best_model = None
        best_map = 0.0

        for model_name, metrics in comparison_results.items():
            if metrics and metrics['mAP50'] > best_map:
                best_map = metrics['mAP50']
                best_model = model_name

        comparison_results['best_model'] = {
            'name': best_model,
            'mAP50': best_map
        }

        return comparison_results

    def _save_evaluation_results(self, model_path, metrics):
        """
        保存评估结果

        Args:
            model_path: 模型路径
            metrics: 评估指标
        """
        try:
            # 确定保存路径
            model_dir = Path(model_path).parent.parent
            results_path = model_dir / 'evaluation_results.json'

            # 添加模型信息
            eval_summary = {
                'model_path': str(model_path),
                'metrics': metrics,
                'timestamp': str(Path(model_path).stat().st_mtime)
            }
            if Path(model_path).exists():  # 确保文件存在再获取时间戳
                eval_summary['timestamp'] = str(Path(model_path).stat().st_mtime)

            # 保存结果
            with open(results_path, 'w') as f:
                json.dump(eval_summary, f, indent=2)

            self.logger.info(f"评估结果已保存到: {results_path}")

        except Exception as e:
            self.logger.warning(f"保存评估结果失败: {e}")

    def _parse_cli_output(self, output_lines):
        """
        解析命令行输出获取评估指标

        Args:
            output_lines: 输出行列表

        Returns:
            指标字典
        """
        metrics = {
            'precision': 0.0,
            'recall': 0.0,
            'mAP50': 0.0,
            'mAP50-95': 0.0
        }

        # 查找包含指标的行
        for line in output_lines:
            if 'all' in line and 'mAP' in line:
                # 解析指标行
                parts = line.split()
                try:
                    # 通常格式: all   640    P    R    mAP50    mAP50-95
                    if len(parts) >= 6:
                        metrics['precision'] = float(parts[2])
                        metrics['recall'] = float(parts[3])
                        metrics['mAP50'] = float(parts[4])
                        metrics['mAP50-95'] = float(parts[5])
                except (ValueError, IndexError):
                    self.logger.warning("无法解析评估指标")

        return metrics

    def calculate_class_distribution(self, dataset_yaml):
        """
        计算数据集中各类别的分布

        Args:
            dataset_yaml: 数据集配置文件路径

        Returns:
            类别分布字典
        """
        import yaml

        with open(dataset_yaml, 'r') as f:
            dataset_config = yaml.safe_load(f)

        # 获取标签目录
        dataset_root = Path(dataset_yaml).parent / dataset_config['path']
        train_labels_dir = dataset_root / 'labels' / 'train'

        # 统计各类别数量
        class_counts = {}
        total_objects = 0

        label_files = list(train_labels_dir.glob('*.txt'))

        for label_file in tqdm(label_files, desc="统计类别分布"):
            with open(label_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.strip():
                        class_id = int(line.split()[0])
                        class_name = dataset_config['names'].get(class_id, f'class_{class_id}')
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1
                        total_objects += 1

        # 计算百分比
        distribution = {}
        for class_name, count in class_counts.items():
            distribution[class_name] = {
                'count': count,
                'percentage': (count / total_objects * 100) if total_objects > 0 else 0
            }

        distribution['total'] = {
            'objects': total_objects,
            'images': len(label_files)
        }

        return distribution