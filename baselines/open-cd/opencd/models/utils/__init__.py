from .builder import build_interaction_layer
from .interaction_layer import (Aggregation_distribution, ChannelExchange,
                                SpatialExchange, TwoIdentity)

# TTP 的时间融合层按需注册，避免 utils 包初始化时拉起额外依赖链。
try:
    from .ttp_layer import TimeFusionTransformerEncoderLayer
except Exception:
    TimeFusionTransformerEncoderLayer = None

__all__ = [
    'build_interaction_layer', 'Aggregation_distribution', 'ChannelExchange', 
    'SpatialExchange', 'TwoIdentity']

if TimeFusionTransformerEncoderLayer is not None:
    __all__.append('TimeFusionTransformerEncoderLayer')
