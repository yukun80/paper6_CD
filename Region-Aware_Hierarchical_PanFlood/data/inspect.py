from typing import Dict

import torch
from torch.utils.data import DataLoader

from data.dataset import UrbanSARFloodsDataset, collate_fn


def inspect_dataset(dataset: UrbanSARFloodsDataset, max_batches: int = 8) -> Dict:
    """统计数据规模、类别分布、输入有效性。"""
    hist = dataset.compute_class_histogram(num_classes=3)
    total = float(hist.sum().item())
    ratios = (hist / max(total, 1.0)).tolist()

    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn)
    nan_count = 0
    b = 0
    for x_dict, _, _ in loader:
        if not torch.isfinite(x_dict["imgs"]).all():
            nan_count += 1
        b += 1
        if b >= max_batches:
            break

    return {
        "num_samples": len(dataset),
        "class_hist": [int(x) for x in hist.tolist()],
        "class_ratio": [float(x) for x in ratios],
        "nan_batch_count": int(nan_count),
        "input_mode": dataset.input_mode,
        "channels": len(dataset.layout),
        "label_mapping_report": dataset.get_label_mapping_report(),
    }
