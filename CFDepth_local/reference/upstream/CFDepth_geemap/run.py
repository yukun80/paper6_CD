#!/usr/bin/env python3
"""CFDepth云端求解、本地调度和GeoTIFF下载命令行。"""

import argparse
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".cache" / "matplotlib"))
from cfdepth.config import Config
from cfdepth.storage import atomic_json


def check_environment():
    import ee, geemap, geedim, torch, torchvision, numpy, scipy, rasterio

    packages = [
        "earthengine-api",
        "geemap",
        "geedim",
        "torch",
        "torchvision",
        "numpy",
        "scipy",
        "rasterio",
    ]
    before = json.loads((HERE / "environment" / "before.json").read_text())
    changed = {
        k: [v, importlib.metadata.version(k)]
        for k, v in before.items()
        if importlib.metadata.version(k) != v
    }
    check = subprocess.run(
        [sys.executable, "-m", "pip", "check"], capture_output=True, text=True
    )
    report = {
        "python": sys.executable,
        "versions": {k: importlib.metadata.version(k) for k in packages},
        "changed_existing_packages": changed,
        "pip_check_ok": check.returncode == 0,
        "credential_file_present": (
            Path.home() / ".config/earthengine/credentials"
        ).is_file(),
    }
    atomic_json(HERE / "environment" / "check.json", report)
    print(json.dumps(report, indent=2))
    if changed or check.returncode:
        raise RuntimeError("Environment consistency failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check-env")
    auth = sub.add_parser("auth")
    auth.add_argument(
        "--auth-mode", default="notebook", choices=["notebook", "localhost", "gcloud"]
    )
    for name in ("check", "run"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        if name == "run":
            p.add_argument("--resume", action="store_true")
            p.add_argument(
                "--stop-after-step",
                type=int,
                help="验收断点：已完成并核验该step后退出；不取消云端任务",
            )
    args = parser.parse_args()
    if args.command == "check-env":
        check_environment()
        return
    if args.command == "auth":
        import ee

        ee.Authenticate(auth_mode=args.auth_mode)
        return
    cfg = Config.read(args.config)
    handlers = [logging.StreamHandler()]
    if args.command == "run":
        cfg.run_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(cfg.run_dir / "run.log"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )
    from cfdepth.backend import Backend

    backend = Backend(cfg)
    report = backend.initialize()
    if args.command == "check":
        report.update(backend.check())
        print(json.dumps(report, indent=2))
        return
    from cfdepth.scheduler import Scheduler

    Scheduler(cfg, backend).run(args.resume, args.stop_after_step)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "Stopped locally. Submitted cloud tasks continue; use --resume.",
            file=sys.stderr,
        )
        sys.exit(130)
    except Exception as exc:
        logging.error("%s: %s", type(exc).__name__, exc)
        sys.exit(1)
