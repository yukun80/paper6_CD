# Copyright (c) Open-CD. All rights reserved.
from .dual_input_encoder_decoder import DIEncoderDecoder
from .siamencoder_decoder import SiamEncoderDecoder
from .siamencoder_multidecoder import SiamEncoderMultiDecoder
from .ban import BAN
from .mtkd import (DistillSiamEncoderDecoder, 
                   DistillSiamEncoderDecoder_ChangeStar, 
                   DistillDIEncoderDecoder, DistillBAN)

# TTP 依赖单独隔离，避免非 TTP 模型在导入 detector 包时触发其可选依赖。
try:
    from .ttp import TimeTravellingPixels
except Exception:
    TimeTravellingPixels = None

__all__ = ['SiamEncoderDecoder', 'DIEncoderDecoder', 'SiamEncoderMultiDecoder',
           'BAN', 'DistillSiamEncoderDecoder',
           'DistillSiamEncoderDecoder_ChangeStar', 'DistillDIEncoderDecoder',
           'DistillBAN']

if TimeTravellingPixels is not None:
    __all__.append('TimeTravellingPixels')
