import json
import logging
import os
from datetime import datetime
from typing import Dict, Optional


def setup_logger(output_dir: str, name: str = "rahp", level: int = logging.INFO) -> logging.Logger:
    """创建控制台+文件双通道日志器。"""
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(os.path.join(output_dir, "train.log"), encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


class JsonlLogger:
    """将训练过程指标追加为 jsonl，便于后处理和可视化。"""

    def __init__(self, output_file: str):
        self.output_file = output_file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    def log(self, payload: Dict, step: Optional[int] = None) -> None:
        data = dict(payload)
        data["timestamp"] = datetime.utcnow().isoformat() + "Z"
        if step is not None:
            data["step"] = int(step)
        with open(self.output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
