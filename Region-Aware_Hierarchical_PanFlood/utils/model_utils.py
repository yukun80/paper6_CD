from typing import Dict

import torch


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """统计模型参数量（总量与可训练量）。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


def summarize_module_parameters(model: torch.nn.Module) -> Dict[str, Dict[str, int]]:
    """输出关键模块参数统计，便于实验记录与复现。"""

    def _cnt(module: torch.nn.Module) -> Dict[str, int]:
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return {"total": int(total), "trainable": int(trainable)}

    summary: Dict[str, Dict[str, int]] = {"model": _cnt(model)}

    # 顶层模块
    for name, module in model.named_children():
        summary[name] = _cnt(module)

    # 关键子模块（若存在）
    key_paths = [
        ("backbone.backbone", ["backbone", "backbone"]),
        ("backbone.patch_embed", ["backbone", "backbone", "patch_embed"]),
        ("backbone.source_role_embed", ["backbone", "source_role_embed"]),
        ("state_memory", ["state_memory"]),
        ("prompt_refiner", ["prompt_refiner"]),
    ]
    for key, path in key_paths:
        cur = model
        ok = True
        for p in path:
            if not hasattr(cur, p):
                ok = False
                break
            cur = getattr(cur, p)
        if ok and isinstance(cur, torch.nn.Module):
            summary[key] = _cnt(cur)
    return summary
