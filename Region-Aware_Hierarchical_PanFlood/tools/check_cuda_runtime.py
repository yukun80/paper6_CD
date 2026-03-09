#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import torch

_THIS = Path(__file__).resolve()
_PROJ = _THIS.parents[1]
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

from utils.runtime import build_device_report, resolve_device


def parse_args():
    parser = argparse.ArgumentParser("Check CUDA runtime for Region-Aware Hierarchical PanFlood")
    parser.add_argument("--device", default="cuda", choices=["auto", "cuda", "cpu"], type=str)
    parser.add_argument("--strict", action="store_true", help="若开启，任一 CUDA 检查失败直接返回非零退出码")
    parser.add_argument("--output", default="", type=str, help="可选 JSON 输出路径")
    return parser.parse_args()


def main():
    args = parse_args()
    report = build_device_report()
    status = {"resolve_device_ok": False, "cuda_matmul_ok": False, "cuda_amp_ok": False, "error": ""}

    try:
        device, _ = resolve_device(requested=args.device, require_cuda=(args.device == "cuda"))
        status["resolve_device_ok"] = True
    except Exception as e:
        status["error"] = str(e)
        device = torch.device("cpu")

    if device.type == "cuda":
        try:
            x = torch.randn((1024, 1024), device=device)
            y = torch.randn((1024, 1024), device=device)
            z = x @ y
            _ = float(z.mean().item())
            status["cuda_matmul_ok"] = True
        except Exception as e:
            status["error"] = str(e)

        try:
            with torch.autocast("cuda", dtype=torch.float16, enabled=True):
                x = torch.randn((512, 512), device=device)
                y = torch.randn((512, 512), device=device)
                z = x @ y
                _ = float(z.mean().item())
            status["cuda_amp_ok"] = True
        except Exception as e:
            status["error"] = str(e)

    out = {"device_report": report, "status": status}
    print(json.dumps(out, ensure_ascii=False, indent=2))

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True) if os.path.dirname(args.output) else None
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    if args.strict:
        ok = status["resolve_device_ok"] and status["cuda_matmul_ok"] and status["cuda_amp_ok"]
        if not ok:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
