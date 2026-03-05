# Copyright (c) Open-CD. All rights reserved.
from opencd.registry import DATASETS
from .basecddataset import _BaseCDDataset


@DATASETS.register_module()
class UrbanSARFloods_CD_Dataset(_BaseCDDataset):
    """UrbanSARFloods change detection dataset (3-class labels)."""

    METAINFO = dict(
        classes=('background_or_no_flood', 'flood_class_1', 'flood_class_2'),
        palette=[[0, 0, 0], [0, 128, 255], [255, 0, 0]])

    def __init__(self,
                 img_suffix='.npy',
                 seg_map_suffix='.tif',
                 format_seg_map=None,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            format_seg_map=format_seg_map,
            **kwargs)
