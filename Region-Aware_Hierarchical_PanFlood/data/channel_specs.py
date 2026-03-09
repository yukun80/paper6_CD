from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ChannelSpec:
    """单通道语义定义。"""

    name: str
    measurement_type: int  # 0 coherence, 1 intensity, 2 delta
    temporal_role: int  # 0 pre, 1 co, 2 post, 3 pre_minus_co, 4 post_minus_pre
    polarization: int  # 0 vv, 1 vh
    source_role: int  # 0 raw, 1 engineered


# 8通道（由 tile_report.json 实测确认）：
# 1 pre_coh_vh, 2 pre_coh_vv, 3 co_coh_vh, 4 co_coh_vv,
# 5 pre_int_vh, 6 pre_int_vv, 7 co_int_vh, 8 co_int_vv
RAW_8CH: List[ChannelSpec] = [
    ChannelSpec("coh_pre_vh", 0, 0, 1, 0),
    ChannelSpec("coh_pre_vv", 0, 0, 0, 0),
    ChannelSpec("coh_co_vh", 0, 1, 1, 0),
    ChannelSpec("coh_co_vv", 0, 1, 0, 0),
    ChannelSpec("int_pre_vh", 1, 0, 1, 0),
    ChannelSpec("int_pre_vv", 1, 0, 0, 0),
    ChannelSpec("int_co_vh", 1, 1, 1, 0),
    ChannelSpec("int_co_vv", 1, 1, 0, 0),
]

# 12通道（由 tile_report.json 实测确认）
RAW_12CH: List[ChannelSpec] = [
    ChannelSpec("coh_pre_vh", 0, 0, 1, 0),
    ChannelSpec("coh_pre_vv", 0, 0, 0, 0),
    ChannelSpec("coh_co_vh", 0, 1, 1, 0),
    ChannelSpec("coh_co_vv", 0, 1, 0, 0),
    ChannelSpec("int_pre_vh", 1, 0, 1, 0),
    ChannelSpec("int_pre_vv", 1, 0, 0, 0),
    ChannelSpec("int_co_vh", 1, 1, 1, 0),
    ChannelSpec("int_co_vv", 1, 1, 0, 0),
    ChannelSpec("dcoh_vh", 2, 3, 1, 0),
    ChannelSpec("dcoh_vv", 2, 3, 0, 0),
    ChannelSpec("dint_vh", 2, 4, 1, 0),
    ChannelSpec("dint_vv", 2, 4, 0, 0),
]

# 工程增强通道：与需求对齐。
# 8ch数据没有显式 post，这里以 co-event 作为 post 近似状态（README 会写明）。
ENGINEERED_4CH: List[ChannelSpec] = [
    ChannelSpec("dint_vv_eng", 2, 4, 0, 1),
    ChannelSpec("dint_vh_eng", 2, 4, 1, 1),
    ChannelSpec("dcoh_vv_eng", 2, 3, 0, 1),
    ChannelSpec("dcoh_vh_eng", 2, 3, 1, 1),
]


def build_channel_layout(mode: str) -> List[ChannelSpec]:
    """根据输入模式生成通道布局定义。"""
    mode = str(mode)
    if mode == "8ch":
        return list(RAW_8CH)
    if mode == "8ch_plus_engineered":
        return list(RAW_8CH) + list(ENGINEERED_4CH)
    if mode == "12ch":
        return list(RAW_12CH)
    raise ValueError(f"Unsupported input mode: {mode}")


def build_channel_id_arrays(layout: List[ChannelSpec]) -> Dict[str, List[int]]:
    """导出 backbone 所需的通道元信息 id 数组。"""
    # chn_ids 约定：SAR 用负数，保持与 panopticon 兼容。
    # 这里用 -1(vv) / -2(vh) 的简化映射，不依赖轨道号。
    chn_ids = [-2 if x.polarization == 1 else -1 for x in layout]
    # 为兼容 Panopticon time_embed，我们让 temporal_role 直接作为 time_ids。
    time_ids = [x.temporal_role for x in layout]
    feature_type_ids = [x.measurement_type for x in layout]
    temporal_role_ids = [x.temporal_role for x in layout]
    polarization_ids = [x.polarization for x in layout]
    source_role_ids = [x.source_role for x in layout]

    return {
        "chn_ids": chn_ids,
        "time_ids": time_ids,
        "feature_type_ids": feature_type_ids,
        "temporal_role_ids": temporal_role_ids,
        "polarization_ids": polarization_ids,
        "source_role_ids": source_role_ids,
        "channel_names": [x.name for x in layout],
    }
