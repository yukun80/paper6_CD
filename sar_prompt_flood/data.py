"""GF3_Henan 预处理 tile 数据读取。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class GF3TileRecord:
    """单个 GF3 tile 的清单记录。"""

    tile_id: str
    npz_path: Path
    row_off: int
    col_off: int
    height: int
    width: int
    valid_ratio: float
    low_confidence: bool


class GF3TileDataset(Dataset):
    """读取 GF3_Henan 预处理后的 tile 数据。"""

    def __init__(self, processed_root: str, max_tiles: int = -1) -> None:
        self.processed_root = Path(processed_root).resolve()
        manifest_path = self.processed_root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest not found: {manifest_path}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.reference = manifest["reference"]
        self.records: List[GF3TileRecord] = []
        for item in manifest["tiles"]:
            self.records.append(
                GF3TileRecord(
                    tile_id=item["tile_id"],
                    npz_path=(self.processed_root / item["npz"]).resolve(),
                    row_off=int(item["row_off"]),
                    col_off=int(item["col_off"]),
                    height=int(item["height"]),
                    width=int(item["width"]),
                    valid_ratio=float(item["valid_ratio"]),
                    low_confidence=bool(item.get("low_confidence", False)),
                )
            )
        if int(max_tiles) > 0:
            self.records = self.records[: int(max_tiles)]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict:
        rec = self.records[index]
        data = np.load(rec.npz_path)
        sample = {
            "tile_id": rec.tile_id,
            "pre": torch.from_numpy(data["pre"].astype(np.float32)),
            "post": torch.from_numpy(data["post"].astype(np.float32)),
            "diff": torch.from_numpy(data["diff"].astype(np.float32)),
            "change_score": torch.from_numpy(data["change_score"].astype(np.float32)),
            "log_ratio_like": torch.from_numpy(data["log_ratio_like"].astype(np.float32)),
            "valid_mask": torch.from_numpy(data["valid_mask"].astype(bool)),
            "meta": {
                "tile_id": rec.tile_id,
                "row_off": rec.row_off,
                "col_off": rec.col_off,
                "height": rec.height,
                "width": rec.width,
                "valid_ratio": rec.valid_ratio,
                "low_confidence": rec.low_confidence,
            },
        }
        return sample


def build_gf3_tile_dataloader(cfg: Dict) -> DataLoader:
    """构建 GF3 tile DataLoader。"""
    ds = GF3TileDataset(
        processed_root=cfg["data"]["processed_root"],
        max_tiles=cfg["inference"].get("max_tiles", -1),
    )
    return DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=int(cfg["runtime"].get("num_workers", 0)),
        collate_fn=lambda batch: batch[0],
    )
