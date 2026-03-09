import os
from typing import Dict, Tuple

import torch


def build_device_report() -> Dict:
    report = {
        "torch_version": torch.__version__,
        "torch_cuda_compiled": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cudnn_available": bool(torch.backends.cudnn.is_available()),
        "cudnn_enabled": bool(torch.backends.cudnn.enabled),
        "cudnn_version": int(torch.backends.cudnn.version() or 0),
    }
    if torch.cuda.is_available():
        devices = []
        for idx in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(idx)
            devices.append(
                {
                    "index": int(idx),
                    "name": prop.name,
                    "total_memory_mb": int(prop.total_memory // (1024 * 1024)),
                    "compute_capability": f"{prop.major}.{prop.minor}",
                }
            )
        report["devices"] = devices
        report["current_device"] = int(torch.cuda.current_device())
    return report


def resolve_device(requested: str = "auto", require_cuda: bool = False) -> Tuple[torch.device, Dict]:
    req = str(requested).lower()
    report = build_device_report()

    if req == "auto":
        device = torch.device("cuda" if report["cuda_available"] else "cpu")
    elif req in {"cuda", "gpu"}:
        if not report["cuda_available"]:
            raise RuntimeError("Requested CUDA device, but torch.cuda.is_available() is False")
        device = torch.device("cuda")
    elif req == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError(f"Unsupported --device: {requested}")

    if require_cuda and device.type != "cuda":
        raise RuntimeError("runtime.require_cuda=true, but selected device is not CUDA")
    if require_cuda and not report["cuda_available"]:
        raise RuntimeError("runtime.require_cuda=true, but torch CUDA runtime is unavailable")
    return device, report


def maybe_disable_xformers(disable: bool) -> None:
    if disable:
        os.environ["XFORMERS_DISABLED"] = "1"
