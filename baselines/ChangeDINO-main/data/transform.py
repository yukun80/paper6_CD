import random

import torchvision.transforms.functional as TF
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class Transforms(object):
    """面向双时相变化检测的轻量增强，支持默认光学和 SAR 模式。"""

    def __init__(self, input_size=256, dataset_mode="default"):
        self.input_size = input_size
        self.dataset_mode = dataset_mode

    def __call__(self, _data):
        img1, img2, cd_label = _data["img1"], _data["img2"], _data["cd_label"]

        if random.random() < 0.5:
            img1 = TF.hflip(img1)
            img2 = TF.hflip(img2)
            cd_label = TF.hflip(cd_label)

        if random.random() < 0.5:
            img1 = TF.vflip(img1)
            img2 = TF.vflip(img2)
            cd_label = TF.vflip(cd_label)

        if random.random() < 0.5:
            angles = [90, 180, 270]
            angle = random.choice(angles)
            img1 = TF.rotate(img1, angle)
            img2 = TF.rotate(img2, angle)
            cd_label = TF.rotate(cd_label, angle, interpolation=InterpolationMode.NEAREST)

        if random.random() < 0.5:
            if self.dataset_mode == "sar":
                # SAR 灰度图避免使用饱和度抖动，仅做轻微亮度/对比扰动。
                jitter_ops = []
                brightness_factor = random.uniform(0.9, 1.1)
                jitter_ops.append(
                    Lambda(lambda img: TF.adjust_brightness(img, brightness_factor))
                )
                contrast_factor = random.uniform(0.9, 1.1)
                jitter_ops.append(
                    Lambda(lambda img: TF.adjust_contrast(img, contrast_factor))
                )
            else:
                jitter_ops = []
                brightness_factor = random.uniform(0.75, 1.25)
                jitter_ops.append(
                    Lambda(lambda img: TF.adjust_brightness(img, brightness_factor))
                )
                contrast_factor = random.uniform(0.75, 1.25)
                jitter_ops.append(
                    Lambda(lambda img: TF.adjust_contrast(img, contrast_factor))
                )
                saturation_factor = random.uniform(0.75, 1.25)
                jitter_ops.append(
                    Lambda(lambda img: TF.adjust_saturation(img, saturation_factor))
                )

            random.shuffle(jitter_ops)
            colorjitter = Compose(jitter_ops)
            img1 = colorjitter(img1)
            img2 = colorjitter(img2)

        if random.random() < 0.5:
            i, j, h, w = transforms.RandomResizedCrop(size=(self.input_size, self.input_size)).get_params(
                img=img1,
                scale=[0.333, 1.0],
                ratio=[0.75, 1.333],
            )
            resize_size = (self.input_size, self.input_size)
            img1 = TF.resized_crop(
                img1, i, j, h, w, size=resize_size, interpolation=InterpolationMode.BILINEAR
            )
            img2 = TF.resized_crop(
                img2, i, j, h, w, size=resize_size, interpolation=InterpolationMode.BILINEAR
            )
            cd_label = TF.resized_crop(
                cd_label, i, j, h, w, size=resize_size, interpolation=InterpolationMode.NEAREST
            )

        return {"img1": img1, "img2": img2, "cd_label": cd_label}


class Lambda(object):
    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


