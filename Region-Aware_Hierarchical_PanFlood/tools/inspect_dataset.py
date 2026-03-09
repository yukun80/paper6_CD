#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_PROJ = _THIS.parents[1]
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

from data.dataset import UrbanSARFloodsDataset
from data.inspect import inspect_dataset
from utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser("Inspect UrbanSARFloods dataset")
    parser.add_argument("--config-file", required=True, type=str)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--output", default="", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config_file)

    root = str(_PROJ.parent)
    data_cfg = cfg["data"]
    split_map = {
        "train": data_cfg["train_split"],
        "val": data_cfg.get("val_split", data_cfg["train_split"]),
        "test": data_cfg.get("test_split", data_cfg.get("val_split", data_cfg["train_split"])),
    }

    ds = UrbanSARFloodsDataset(
        data_root=os.path.normpath(os.path.join(root, data_cfg["root"])),
        split_file=split_map[args.split],
        input_mode=data_cfg["input_mode"],
        mean=data_cfg["mean"],
        std=data_cfg["std"],
        crop_size=data_cfg.get("crop_size", 252),
        random_crop=False,
        random_hflip=False,
        random_vflip=False,
        ignore_index=data_cfg.get("ignore_index", 255),
        auto_label_mapping=data_cfg.get("auto_label_mapping", False),
        seed=cfg.get("seed", 42),
    )
    report = inspect_dataset(ds)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
