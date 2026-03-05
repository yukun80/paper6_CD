from collections.abc import Sequence
import os.path as osp
import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC

from mmseg.datasets.builder import PIPELINES

@PIPELINES.register_module()
class wc_LoadImageFromFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self,
                 file_client_args=dict(backend='disk')):

        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        if results.get('aux_prefix') is not None:
            aux_filename = osp.join(results['aux_prefix'],
                                    results['img_info']['aux_filename'])
        else:
            aux_filename = results['img_info']['aux_filename']

        img = mmcv.imread(filename, 'unchanged')
        if len(img.shape) < 3:
            img = np.expand_dims(img, -1)
        if img.dtype == np.float32:
            img = np.clip(img, 0.0, 1.0)
        aux = mmcv.imread(aux_filename, 'unchanged')
        if len(aux.shape) < 3:
            aux = np.expand_dims(aux, -1)
        if aux.dtype == np.float32:
            aux = np.clip(aux, 0.0, 1.0)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['aux'] = aux
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32))
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class wc_LoadNpyFromFile(object):
    """Load 2-temporal npy arrays for water change detection.

    The expected npy shape is ``(C, H, W)`` and it will be converted to
    ``(H, W, C)`` for the downstream pipeline.
    """

    def __init__(self, allow_nan=False, nan_fill_value=0.0):
        self.allow_nan = allow_nan
        self.nan_fill_value = float(nan_fill_value)

    def __call__(self, results):
        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'], results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        if results.get('aux_prefix') is not None:
            aux_filename = osp.join(results['aux_prefix'], results['img_info']['aux_filename'])
        else:
            aux_filename = results['img_info']['aux_filename']

        raw_img = np.load(filename)
        raw_aux = np.load(aux_filename)

        if raw_img.ndim == 2:
            raw_img = raw_img[np.newaxis, ...]
        if raw_aux.ndim == 2:
            raw_aux = raw_aux[np.newaxis, ...]
        if raw_img.ndim != 3 or raw_aux.ndim != 3:
            raise ValueError(f'Expected 3D arrays, got img={raw_img.shape}, aux={raw_aux.shape}')

        invalid_mask = (~np.isfinite(raw_img)).any(axis=0) | (~np.isfinite(raw_aux)).any(axis=0)

        img = np.moveaxis(raw_img, 0, -1).astype(np.float32, copy=False)
        aux = np.moveaxis(raw_aux, 0, -1).astype(np.float32, copy=False)
        if not self.allow_nan:
            img = np.nan_to_num(
                img, nan=self.nan_fill_value, posinf=self.nan_fill_value, neginf=self.nan_fill_value)
            aux = np.nan_to_num(
                aux, nan=self.nan_fill_value, posinf=self.nan_fill_value, neginf=self.nan_fill_value)

        if img.shape[:2] != aux.shape[:2]:
            raise ValueError(f'Shape mismatch between img and aux: {img.shape} vs {aux.shape}')

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['aux'] = aux
        results['invalid_mask'] = invalid_mask.astype(np.uint8)
        if 'seg_fields' not in results:
            results['seg_fields'] = []
        if 'invalid_mask' not in results['seg_fields']:
            results['seg_fields'].append('invalid_mask')
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32))
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(allow_nan={self.allow_nan}, nan_fill_value={self.nan_fill_value})'


@PIPELINES.register_module()
class wc_Normalize(object):
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """

        results['img'] = results['img'].astype(np.float32) / 255.0

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class wc_Standardize(object):
    """Standardize image by channel with mean/std."""

    def __init__(self, mean, std, eps=1e-6):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.eps = float(eps)
        if self.mean.ndim != 1 or self.std.ndim != 1:
            raise ValueError('mean/std should be 1D sequences.')
        if self.mean.shape[0] != self.std.shape[0]:
            raise ValueError('mean and std must have the same length.')

    def __call__(self, results):
        img = results['img'].astype(np.float32, copy=False)
        if img.shape[-1] != self.mean.shape[0]:
            raise ValueError(
                f'Channel mismatch: img has {img.shape[-1]} channels but mean/std has {self.mean.shape[0]}')
        results['img'] = (img - self.mean) / np.maximum(self.std, self.eps)
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(channels={self.mean.shape[0]}, eps={self.eps})'


@PIPELINES.register_module()
class wc_MaskInvalidPixels(object):
    """Mask invalid pixels from input (NaN/Inf) with ignore index in GT."""

    def __init__(self, ignore_index=255):
        self.ignore_index = int(ignore_index)

    def __call__(self, results):
        if 'invalid_mask' not in results or 'gt_semantic_seg' not in results:
            return results
        invalid = results['invalid_mask']
        if invalid.ndim == 3:
            invalid = invalid.squeeze(-1)
        gt = results['gt_semantic_seg']
        if invalid.shape != gt.shape:
            raise ValueError(f'invalid_mask shape {invalid.shape} mismatches gt {gt.shape}')
        gt[invalid > 0] = self.ignore_index
        results['gt_semantic_seg'] = gt
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(ignore_index={self.ignore_index})'


@PIPELINES.register_module()
class wc_StackByChannel(object):
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        arr_list = [None] * len(self.keys)
        for i, key in enumerate(self.keys):
            arr_list[i] = results[key]
        img = np.dstack(arr_list)
        results['img'] = img
        results['img_shape'] = img.shape
        return results

@PIPELINES.register_module()
class wc_SelectChannels(object):
    def __init__(self, key, channels):
        self.key = key
        self.channels = channels

    def __call__(self, results):
        img = results[self.key]
        img = img[..., self.channels]
        results[self.key] = img
        results['{}_shape'.format(self.key)] = img.shape
        return results


@PIPELINES.register_module()
class wc_PhotoMetricDistortion(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def convert(self, img, alpha=1, beta=0):
        """Multiple with alpha and add beat with clip."""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        """Brightness distortion."""
        if random.randint(2):
            return self.convert(
                img,
                beta=random.uniform(-self.brightness_delta,
                                    self.brightness_delta))
        return img

    def contrast(self, img):
        """Contrast distortion."""
        if random.randint(2):
            return self.convert(
                img,
                alpha=random.uniform(self.contrast_lower, self.contrast_upper))
        return img


    def __call__(self, results):
        """Call function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """

        img = results['img']
        # random brightness
        img = self.brightness(img)

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            img = self.contrast(img)

        # random contrast
        if mode == 0:
            img = self.contrast(img)

        results['img'] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(brightness_delta={self.brightness_delta}, '
                     f'contrast_range=({self.contrast_lower}, '
                     f'{self.contrast_upper})')
        return repr_str
