# Copyright (c) Open-CD. All rights reserved.
import io
import warnings
from typing import Dict, Optional, Tuple, Union

import mmcv
import mmengine.fileio as fileio
import numpy as np
from mmcv.transforms import BaseTransform
from mmcv.transforms import LoadAnnotations as MMCV_LoadAnnotations
from mmcv.transforms import LoadImageFromFile as MMCV_LoadImageFromFile

from opencd.registry import TRANSFORMS


@TRANSFORMS.register_module()
class MultiImgLoadImageFromFile(MMCV_LoadImageFromFile):
    """Load an image pair from files.

    Required Keys:

    - img_path

    Modified Keys:

    - img
    - img_shape
    - ori_shape

    """

    def __init__(self, **kwargs) -> None:
         super().__init__(**kwargs)

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filenames = results['img_path']
        imgs = []
        try:
            for filename in filenames:
                if self.file_client_args is not None:
                    file_client = fileio.FileClient.infer_client(
                        self.file_client_args, filename)
                    img_bytes = file_client.get(filename)
                else:
                    img_bytes = fileio.get(
                        filename, backend_args=self.backend_args)
                img = mmcv.imfrombytes(
                img_bytes, flag=self.color_type, backend=self.imdecode_backend)
                if self.to_float32:
                    img = img.astype(np.float32)
                imgs.append(img)
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        
        results['img'] = imgs
        results['img_shape'] = imgs[0].shape[:2]
        results['ori_shape'] = imgs[0].shape[:2]
        return results


@TRANSFORMS.register_module()
class MultiImgLoadNpyFromFile(BaseTransform):
    """Load an image pair from npy files.

    Each npy image is expected to be in (C, H, W). This transform will convert
    it to (H, W, C) to keep the same convention as image loaders.

    Args:
        to_float32 (bool): Whether to cast loaded arrays to float32.
        sanitize_non_finite (bool): Whether to replace NaN/Inf values.
        nan_fill (float): Fill value for NaN.
        posinf_fill (float): Fill value for +Inf.
        neginf_fill (float): Fill value for -Inf.
        expected_channels (int, optional): If set, enforce channel count.
        expected_hw (tuple[int, int], optional): If set, enforce (H, W).
        enforce_pair_shape (bool): Whether all images in one sample must share
            the same spatial shape.
    """

    def __init__(self,
                 to_float32: bool = True,
                 sanitize_non_finite: bool = True,
                 nan_fill: float = 0.0,
                 posinf_fill: float = 0.0,
                 neginf_fill: float = 0.0,
                 expected_channels: Optional[int] = None,
                 expected_hw: Optional[Tuple[int, int]] = None,
                 enforce_pair_shape: bool = True) -> None:
        self.to_float32 = to_float32
        self.sanitize_non_finite = sanitize_non_finite
        self.nan_fill = nan_fill
        self.posinf_fill = posinf_fill
        self.neginf_fill = neginf_fill
        self.expected_channels = expected_channels
        self.expected_hw = expected_hw
        self.enforce_pair_shape = enforce_pair_shape

    def transform(self, results: dict) -> Optional[dict]:
        filenames = results['img_path']
        imgs = []
        invalid_mask = None
        pair_hw = None
        try:
            for filename in filenames:
                img_bytes = fileio.get(filename, backend_args=None)
                img = np.load(io.BytesIO(img_bytes), allow_pickle=False)
                if img.ndim != 3:
                    raise ValueError(
                        f'Expected npy shape (C,H,W), but got {img.shape} '
                        f'for file: {filename}')
                if self.expected_channels is not None and \
                        img.shape[0] != self.expected_channels:
                    raise ValueError(
                        f'Expected {self.expected_channels} channels, got '
                        f'{img.shape[0]} for file: {filename}')

                current_hw = tuple(img.shape[1:])
                if self.expected_hw is not None and current_hw != self.expected_hw:
                    raise ValueError(
                        f'Expected spatial shape {self.expected_hw}, got '
                        f'{current_hw} for file: {filename}')
                if pair_hw is None:
                    pair_hw = current_hw
                elif self.enforce_pair_shape and current_hw != pair_hw:
                    raise ValueError(
                        f'Image pair spatial mismatch: previous {pair_hw}, '
                        f'current {current_hw} for file: {filename}')

                non_finite = ~np.isfinite(img)
                pixel_invalid = np.any(non_finite, axis=0)
                if invalid_mask is None:
                    invalid_mask = pixel_invalid
                else:
                    invalid_mask |= pixel_invalid
                if self.sanitize_non_finite and not np.isfinite(img).all():
                    img = np.nan_to_num(
                        img,
                        nan=self.nan_fill,
                        posinf=self.posinf_fill,
                        neginf=self.neginf_fill)
                img = np.transpose(img, (1, 2, 0))
                if self.to_float32:
                    img = img.astype(np.float32, copy=False)
                imgs.append(img)
        except Exception as e:
            raise e

        if len(imgs) == 0:
            raise ValueError('No image loaded for `MultiImgLoadNpyFromFile`.')

        results['img'] = imgs
        results['img_shape'] = imgs[0].shape[:2]
        results['ori_shape'] = imgs[0].shape[:2]
        if invalid_mask is not None:
            results['invalid_mask'] = invalid_mask.astype(np.uint8, copy=False)
        return results


@TRANSFORMS.register_module()
class MultiImgApplyInvalidMask(BaseTransform):
    """Mask invalid pixels in ``gt_seg_map`` with ``ignore_index``.

    This transform expects ``invalid_mask`` generated by
    ``MultiImgLoadNpyFromFile``. Invalid pixels are excluded from loss by
    setting corresponding labels to ``ignore_index``.
    """

    def __init__(self, ignore_index: int = 255, drop_invalid_mask: bool = True):
        self.ignore_index = ignore_index
        self.drop_invalid_mask = drop_invalid_mask

    def transform(self, results: dict) -> dict:
        if 'invalid_mask' not in results or 'gt_seg_map' not in results:
            return results

        invalid_mask = results['invalid_mask']
        gt_seg_map = results['gt_seg_map']
        if invalid_mask.shape != gt_seg_map.shape:
            raise ValueError(
                f'`invalid_mask` shape {invalid_mask.shape} does not match '
                f'`gt_seg_map` shape {gt_seg_map.shape}.')

        gt_seg_map = gt_seg_map.copy()
        gt_seg_map[invalid_mask > 0] = self.ignore_index
        results['gt_seg_map'] = gt_seg_map
        results['invalid_ratio'] = float((invalid_mask > 0).mean())

        if self.drop_invalid_mask:
            results.pop('invalid_mask', None)
        return results


@TRANSFORMS.register_module()
class MultiImgLoadAnnotations(MMCV_LoadAnnotations):
    """Load annotations for change detection provided by dataset.

    The annotation format is as the following:

    .. code-block:: python

        {
            # Filename of change detection ground truth file.
            'seg_map_path': 'a/b/c'
        }

    After this module, the annotation has been changed to the format below:

    .. code-block:: python

        {
            # in str
            'seg_fields': List
             # In uint8 type.
            'gt_seg_map': np.ndarray (H, W)
        }

    Required Keys:

    - seg_map_path (str): Path of change detection ground truth file.

    Added Keys:

    - seg_fields (List)
    - gt_seg_map (np.uint8)

    Args:
        reduce_zero_label (bool, optional): Whether reduce all label value
            by 255. Usually used for datasets where 0 is background label.
            Defaults to None.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:``mmcv.imfrombytes``.
            See :fun:``mmcv.imfrombytes`` for details.
            Defaults to 'pillow'.
        backend_args (dict): Arguments to instantiate a file backend.
            See https://mmengine.readthedocs.io/en/latest/api/fileio.htm
            for details. Defaults to None.
            Notes: mmcv>=2.0.0rc4, mmengine>=0.2.0 required.
    """

    def __init__(
        self,
        reduce_zero_label=None,
        backend_args=None,
        imdecode_backend='pillow',
    ) -> None:
        super().__init__(
            with_bbox=False,
            with_label=False,
            with_seg=True,
            with_keypoints=False,
            imdecode_backend=imdecode_backend,
            backend_args=backend_args)
        self.reduce_zero_label = reduce_zero_label
        if self.reduce_zero_label is not None:
            warnings.warn('`reduce_zero_label` will be deprecated, '
                          'if you would like to ignore the zero label, please '
                          'set `reduce_zero_label=True` when dataset '
                          'initialized')
        self.imdecode_backend = imdecode_backend

    def _load_seg_map(self, results: dict) -> None:
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        img_bytes = fileio.get(
            results['seg_map_path'], backend_args=self.backend_args)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='grayscale', # in mmseg: unchanged
            backend=self.imdecode_backend).squeeze().astype(np.uint8)

        # reduce zero_label
        if self.reduce_zero_label is None:
            self.reduce_zero_label = results['reduce_zero_label']
        assert self.reduce_zero_label == results['reduce_zero_label'], \
            'Initialize dataset with `reduce_zero_label` as ' \
            f'{results["reduce_zero_label"]} but when load annotation ' \
            f'the `reduce_zero_label` is {self.reduce_zero_label}'
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        # modify to format ann
        if results.get('format_seg_map', None) is not None:
            if results['format_seg_map'] == 'to_binary':
                gt_semantic_seg_copy = gt_semantic_seg.copy()
                gt_semantic_seg[gt_semantic_seg_copy < 128] = 0
                gt_semantic_seg[gt_semantic_seg_copy >= 128] = 1
            else:
                raise ValueError('Invalid value {}'.format(results['format_seg_map']))
        # modify if custom classes
        if results.get('label_map', None) is not None:
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
        results['gt_seg_map'] = gt_semantic_seg
        results['seg_fields'].append('gt_seg_map')

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'backend_args={self.backend_args})'
        return repr_str


@TRANSFORMS.register_module()
class MultiImgMultiAnnLoadAnnotations(MMCV_LoadAnnotations):
    """Load annotations for semantic change detection provided by dataset.

    The annotation format is as the following:

    .. code-block:: python

        {
            # Filename of change detection ground truth file.
            'seg_map_path': 'a/b/c'
        }

    After this module, the annotation has been changed to the format below:

    .. code-block:: python

        {
            # in str
            'seg_fields': List
             # In uint8 type.
            'gt_seg_map': np.ndarray (H, W)
        }

    Required Keys:

    - seg_map_path (str): Path of change detection ground truth file.

    Added Keys:

    - seg_fields (List)
    - gt_seg_map (np.uint8)

    Args:
        reduce_semantic_zero_label (bool, optional): Whether reduce all label value
            by 255. Usually used for datasets where 0 is background label.
            Defaults to None.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:``mmcv.imfrombytes``.
            See :fun:``mmcv.imfrombytes`` for details.
            Defaults to 'pillow'.
        backend_args (dict): Arguments to instantiate a file backend.
            See https://mmengine.readthedocs.io/en/latest/api/fileio.htm
            for details. Defaults to None.
            Notes: mmcv>=2.0.0rc4, mmengine>=0.2.0 required.
    """

    def __init__(
        self,
        reduce_semantic_zero_label=None,
        backend_args=None,
        imdecode_backend='pillow',
    ) -> None:
        super().__init__(
            with_bbox=False,
            with_label=False,
            with_seg=True,
            with_keypoints=False,
            imdecode_backend=imdecode_backend,
            backend_args=backend_args)
        self.reduce_semantic_zero_label = reduce_semantic_zero_label
        if self.reduce_semantic_zero_label is not None:
            warnings.warn('`reduce_semantic_zero_label` will be deprecated, '
                          'if you would like to ignore the zero label, please '
                          'set `reduce_semantic_zero_label=True` when dataset '
                          'initialized')
        self.imdecode_backend = imdecode_backend

    def _load_seg_map(self, results: dict) -> None:
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        img_bytes = fileio.get(
            results['seg_map_path'], backend_args=self.backend_args)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='grayscale', # in mmseg: unchanged
            backend=self.imdecode_backend).squeeze().astype(np.uint8)
        # for semantic anns
        img_bytes_from = fileio.get(
            results['seg_map_path_from'], backend_args=self.backend_args)
        gt_semantic_seg_from = mmcv.imfrombytes(
            img_bytes_from, flag='grayscale',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)
        img_bytes_to = fileio.get(
            results['seg_map_path_to'], backend_args=self.backend_args)
        gt_semantic_seg_to = mmcv.imfrombytes(
            img_bytes_to, flag='grayscale',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)

        # reduce zero_label
        if self.reduce_semantic_zero_label is None:
            self.reduce_semantic_zero_label = results['reduce_semantic_zero_label']
        assert self.reduce_semantic_zero_label == results['reduce_semantic_zero_label'], \
            'Initialize dataset with `reduce_semantic_zero_label` as ' \
            f'{results["reduce_semantic_zero_label"]} but when load annotation ' \
            f'the `reduce_semantic_zero_label` is {self.reduce_semantic_zero_label}'
        if self.reduce_semantic_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg_from[gt_semantic_seg_from == 0] = 255
            gt_semantic_seg_from = gt_semantic_seg_from - 1
            gt_semantic_seg_from[gt_semantic_seg_from == 254] = 255
            gt_semantic_seg_to[gt_semantic_seg_to == 0] = 255
            gt_semantic_seg_to = gt_semantic_seg_to - 1
            gt_semantic_seg_to[gt_semantic_seg_to == 254] = 255
        # modify to format ann
        if results.get('format_seg_map', None) is not None:
            if results['format_seg_map'] == 'to_binary':
                gt_semantic_seg_copy = gt_semantic_seg.copy()
                gt_semantic_seg[gt_semantic_seg_copy < 128] = 0
                gt_semantic_seg[gt_semantic_seg_copy >= 128] = 1
            else:
                raise ValueError('Invalid value {}'.format(results['format_seg_map']))
        # modify if custom classes
        if results.get('label_map', None) is not None:
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
        if results.get('semantic_label_map', None) is not None:
            ''' Just for semantic anns here '''
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_from_copy = gt_semantic_seg_from.copy()
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg_from[gt_semantic_seg_from_copy == old_id] = new_id
            gt_semantic_seg_to_copy = gt_semantic_seg_to.copy()
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg_to[gt_semantic_seg_to_copy == old_id] = new_id

        results['gt_seg_map'] = gt_semantic_seg
        results['gt_seg_map_from'] = gt_semantic_seg_from
        results['gt_seg_map_to'] = gt_semantic_seg_to
        results['seg_fields'].extend(['gt_seg_map', 
            'gt_seg_map_from', 'gt_seg_map_to'])

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_semantic_zero_label={self.reduce_semantic_zero_label}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'backend_args={self.backend_args})'
        return repr_str


@TRANSFORMS.register_module()
class MultiImgLoadLoadImageFromNDArray(MultiImgLoadImageFromFile):
    """Load an image pair from ``results['img']``.

    Similar with :obj:`LoadImageFromFile`, but the image has been loaded as
    :obj:`np.ndarray` in ``results['img']``. Can be used when loading image
    from webcam.

    Required Keys:

    - img

    Modified Keys:

    - img
    - img_path
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
    """

    def transform(self, results: dict) -> dict:
        """Transform function to add image meta information.

        Args:
            results (dict): Result dict with Webcam read image in
                ``results['img']``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        imgs = []
        for img in results["img"]:
            if self.to_float32:
                img = img.astype(np.float32)
            imgs.append(img)

        results['img_path'] = None
        results['img'] = imgs
        results['img_shape'] = imgs[0].shape[:2]
        results['ori_shape'] = imgs[0].shape[:2]
        return results


@TRANSFORMS.register_module()
class MultiImgLoadInferencerLoader(BaseTransform):
    """Load an image pair from ``results['img']``.

    Similar with :obj:`LoadImageFromFile`, but the image has been loaded as
    :obj:`np.ndarray` in ``results['img']``. Can be used when loading image
    from webcam.

    Required Keys:

    - img

    Modified Keys:

    - img
    - img_path
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.from_file = TRANSFORMS.build(
            dict(type='MultiImgLoadImageFromFile', **kwargs))
        self.from_ndarray = TRANSFORMS.build(
            dict(type='MultiImgLoadLoadImageFromNDArray', **kwargs))

    def transform(self, single_input: Union[str, np.ndarray, dict]) -> dict:
        """Transform function to add image meta information.

        Args:
            results (dict): Result dict with Webcam read image in
                ``results['img']``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        assert len(single_input) == 2, \
            'In `MultiImgLoadInferencerLoader`,' \
            '`single_input` contains bi-temporal images'
        if isinstance(single_input[0], str):
            inputs = dict(img_path=single_input)
        elif isinstance(single_input[0], Union[np.ndarray, list]):
            inputs = dict(img=single_input)
        elif isinstance(single_input[0], dict):
            inputs = single_input
        else:
            raise NotImplementedError

        if 'img' in inputs:
            return self.from_ndarray(inputs)
        return self.from_file(inputs)
