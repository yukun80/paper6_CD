import torch.nn as nn
import copy

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


def build_voxelization(cfg):
    ret = nn.ModuleDict()
    if cfg is None:
        return ret
    cfg_cp = copy.deepcopy(cfg)
    for name in cfg_cp:
        if has(cfg_cp, name):
            ret[name] = MODELS.build(cfg_cp[name])
    return ret


def build_backbone(cfg):
    ret = nn.ModuleDict()
    if cfg is None:
        return ret
    if has(cfg, 'image'):
        ret['image'] = MODELS.build(cfg['image'])

    if has(cfg, 'point'):
        ret['point'] = nn.ModuleDict()
        for name in cfg['point']:
            if has(cfg['point'], name):
                ret['point'][name] = MODELS.build(cfg['point'][name])
    return ret


def build_fusion(cfg):
    if cfg is None:
        return None
    ret = MODELS.build(cfg)
    return ret


def build_head(cfg):
    cfg_cp = copy.deepcopy(cfg)
    ret = nn.ModuleDict()
    for name, module in cfg_cp.items():
        if module is None:
            continue
        ret[name] = MODELS.build(module)
    return ret


class ModelTemplate(nn.Module):
    """pure model template for 3D perception."""

    def __init__(self,
                 voxelization,
                 backbone,
                 fusion,
                 head,
                 **kwargs):
        super(ModelTemplate, self).__init__()
        self.voxelization = build_voxelization(voxelization)
        self.backbone = build_backbone(backbone)
        self.fusion = build_fusion(fusion)
        self.head = build_head(head)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "ModelTemplate is a pure model template, "
            "please implement your own forward function."
        )
