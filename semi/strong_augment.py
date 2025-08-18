# semi/strong_augment.py

import random
import numpy as np
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image

# RandAugment 操作列表和强度参数
PARAMETER_MAX = 10
operations = {
    'AutoContrast': lambda img, _: PIL.ImageOps.autocontrast(img),
    'Equalize': lambda img, _: PIL.ImageOps.equalize(img),
    # 'Invert': lambda img, _: PIL.ImageOps.invert(img),
    # 'Rotate': lambda img, mag: img.rotate(mag * 30),
    'Posterize': lambda img, mag: PIL.ImageOps.posterize(img, 4 - int(mag * 4 / PARAMETER_MAX)),
    # 'Solarize': lambda img, mag: PIL.ImageOps.solarize(img, 256 - int(mag * 255 / PARAMETER_MAX)),
    'Color': lambda img, mag: PIL.ImageEnhance.Color(img).enhance(1 - mag * 0.9),
    'Contrast': lambda img, mag: PIL.ImageEnhance.Contrast(img).enhance(1 - mag * 0.9),
    'Brightness': lambda img, mag: PIL.ImageEnhance.Brightness(img).enhance(1 - mag * 0.9),
    'Sharpness': lambda img, mag: PIL.ImageEnhance.Sharpness(img).enhance(1 - mag * 0.9),
    # 'ShearX': lambda img, mag: img.transform(img.size, PIL.Image.AFFINE, (1, mag * 0.3, 0, 0, 1, 0)),
    # 'ShearY': lambda img, mag: img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, mag * 0.3, 1, 0)),
    'TranslateX': lambda img, mag: img.transform(img.size, PIL.Image.AFFINE, (1, 0, mag * img.size[0] * 0.45, 0, 1, 0)),
    'TranslateY': lambda img, mag: img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, mag * img.size[1] * 0.45)),
}


class RandAugment:
    def __init__(self, n, m):
        self.n = n
        self.m = m

    def __call__(self, img):
        ops = random.choices(list(operations.keys()), k=self.n)
        for op_name in ops:
            magnitude = np.random.uniform(low=0.1, high=self.m)
            img = operations[op_name](img, magnitude)
        return img


class Cutout:
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h, w = img.size[1], img.size[0]
        mask = np.ones((h, w), np.float32)
        for _ in range(self.n_holes):
            y, x = np.random.randint(h), np.random.randint(w)
            y1, y2 = np.clip(y - self.length // 2, 0, h), np.clip(y + self.length // 2, 0, h)
            x1, x2 = np.clip(x - self.length // 2, 0, w), np.clip(x + self.length // 2, 0, w)
            mask[y1:y2, x1:x2] = 0.

        mask = Image.fromarray(mask, 'L')
        img_rgba = img.copy().convert('RGBA')
        img_rgba.putalpha(mask)

        # 返回带有白色背景的RGB图像，以兼容YOLO的数据格式
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img_rgba, mask=img_rgba.split()[3])
        return background


class StrongAugment:
    """组合RandAugment和Cutout的强增强类"""

    def __init__(self, n=2, m=7, cutout_n=0, cutout_len=0):
        self.rand_augment = RandAugment(n, m)
        self.cutout = Cutout(n_holes=cutout_n, length=cutout_len)

    def __call__(self, img):
        img = self.rand_augment(img)
        img = self.cutout(img)
        return img