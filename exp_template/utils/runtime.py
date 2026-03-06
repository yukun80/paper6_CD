"""运行时相关工具。"""

from __future__ import annotations

import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def set_torch_home(project_root: str) -> str:
    """统一预训练权重缓存路径到 exp_template/checkpoints。"""

    torch_home = (Path(project_root) / "exp_template" / "checkpoints").resolve()
    torch_home.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(torch_home)
    return str(torch_home)


def setup_seed(seed: int, deterministic: bool = False) -> None:
    """设置随机种子，保证实验可复现。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def now_str() -> str:
    """返回当前时间字符串，用于 run_name。"""

    return datetime.now().strftime("%Y%m%d_%H%M%S")
