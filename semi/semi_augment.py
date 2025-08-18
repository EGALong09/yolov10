from ultralytics.data.augment import (
    Compose, Mosaic, RandomPerspective, LetterBox,
    MixUp, CopyPaste, Albumentations, RandomHSV, RandomFlip
)
from semi.strong_augment import RandAugment


# 增强部分
def build_strong_transforms(dataset, imgsz, hyp):
    """
    构建标准的强数据增强流水线。
    包含Mosaic, MixUp, 强几何变换等。
    """
    # 这部分代码与YOLOv10默认的v8_transforms基本一致
    # 我们保留Mosaic, RandomPerspective等强几何变换
    pre_transform = Compose([
        Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic),
        RandomPerspective(
            degrees=hyp.degrees,
            translate=hyp.translate,
            scale=hyp.scale,
            shear=hyp.shear,
            perspective=hyp.perspective,
            border=dataset.mosaic_border
        ),
        # MixUp和CopyPaste也是强增强
        MixUp(dataset, p=hyp.mixup),
        CopyPaste(p=hyp.copy_paste),
    ])

    class ClipBboxes:
        def __call__(self, labels):
            bboxes = labels.get("bboxes")
            if bboxes is not None and len(bboxes) > 0:
                # 将所有坐标裁剪到 [0.0, 1.0] 范围内
                # bboxes[:, :4] 对应 x, y, w, h
                np.clip(bboxes[:, :4], 0.0, 1.0, out=bboxes[:, :4])
            labels["bboxes"] = bboxes
            return labels

    return Compose([
        pre_transform,
        # 这里可以加入您自定义的RandAugment，使其成为强增强的一部分
        # RandAugment(p=0.5), # 假设RandAugment接受一个概率参数
        ClipBboxes(),
        Albumentations(p=1.0),
        RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
        RandomFlip(p=hyp.fliplr, direction='horizontal'),
        RandomFlip(p=hyp.flipud, direction='vertical'),
    ])


def build_weak_transforms(dataset, imgsz, hyp):
    """
    构建一个“小目标友好”的弱数据增强流水线。
    这部分等同于YOLOv10的默认增强，只包含色彩变换和翻转，
    不包含任何破坏性的几何变换。
    """
    # 弱增强流水线中，我们只保留最安全、最基础的增强
    return Compose([
        # 关键：完全移除 Mosaic, RandomPerspective, MixUp, CopyPaste
        LetterBox(new_shape=(imgsz, imgsz), auto=False),  # 仅进行缩放和填充
        Albumentations(p=1.0),
        RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
        RandomFlip(p=hyp.fliplr, direction='horizontal'),
        # 通常垂直翻转在交通场景下是不安全的，可以禁用
        # RandomFlip(p=hyp.flipud, direction='vertical'),
    ])