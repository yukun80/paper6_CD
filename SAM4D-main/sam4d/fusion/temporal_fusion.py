import torch.nn as nn
from mmengine.registry import MODELS


def has(cfg, name):
    if name in cfg:
        if cfg[name] is not None:
            if isinstance(cfg[name], list):
                if len(cfg[name]) > 0:
                    return True
            else:
                return True
    return False


def build_sam4d_mem_encoder(cfg):
    ret = nn.ModuleDict()
    if cfg is None:
        return ret
    if has(cfg, 'image'):
        ret['image'] = MODELS.build(cfg['image'])
    if has(cfg, 'point'):
        ret['point'] = MODELS.build(cfg['point'])
    return ret


@MODELS.register_module()
class SAM4DFusion(nn.Module):
    """CenterHead for CenterPoint.

    Args:
        modal_fusion (dict, optional): modal fusion configs.
         Default: None.
        temporal_fusion (dict, optional): temporal fusion configs.
         Default: None.
        before_task (dict | list[dict], optional): configs operations(like conv) before task.
         Default: None.
    """

    def __init__(self,
                 memory_attention=None,
                 memory_encoder=None,
                 **kwargs):
        super(SAM4DFusion, self).__init__()

        if memory_attention is not None:
            self.memory_attention = MODELS.build(memory_attention)

        if memory_encoder is not None:
            self.memory_encoder = build_sam4d_mem_encoder(memory_encoder)

    def forward(self, input_dict):
        raise NotImplementedError("you need to call fusion with memory_attention and memory_encoder by yourself")
