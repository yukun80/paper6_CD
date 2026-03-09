import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """统一随机种子，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
