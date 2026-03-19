from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .transform import Transforms


def _scan_split_dir(split_dir: Path) -> dict[str, Path]:
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    files = sorted(p for p in split_dir.iterdir() if p.is_file())
    return {p.name: p for p in files}


def _resolve_label_dir(base_dir: Path) -> Path:
    candidates = [base_dir / "label", base_dir / "Label"]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Cannot find label directory under: {base_dir}")


def _normalize_to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    if image.mode in {"L", "I", "F", "P", "LA"}:
        return image.convert("RGB")
    return image.convert("RGB")


class Load_Dataset(Dataset):
    """读取标准变化检测目录，并兼容 S1GFloods 的 SAR PNG 输入。"""

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        split_root = Path(opt.dataroot) / opt.dataset / opt.phase
        dir1 = split_root / "A"
        dir2 = split_root / "B"
        dir_label = _resolve_label_dir(split_root)

        self.t1_map = _scan_split_dir(dir1)
        self.t2_map = _scan_split_dir(dir2)
        self.label_map = _scan_split_dir(dir_label)
        self.fnames = sorted(self.t1_map.keys())

        if self.fnames != sorted(self.t2_map.keys()) or self.fnames != sorted(self.label_map.keys()):
            raise ValueError(
                f"File mismatch under split {split_root}: "
                "A/B/label must have identical file names."
            )
        if not self.fnames:
            raise ValueError(f"No samples found under split: {split_root}")

        self.dataset_size = len(self.fnames)
        self.normalize = transforms.Normalize(tuple(opt.mean), tuple(opt.std))
        self.transform = Transforms(input_size=opt.input_size, dataset_mode=opt.dataset_mode)
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        fname = self.fnames[index]
        img1 = _normalize_to_rgb(Image.open(self.t1_map[fname]))
        img2 = _normalize_to_rgb(Image.open(self.t2_map[fname]))

        label = np.array(Image.open(self.label_map[fname]).convert("L"), dtype=np.uint8)
        label = (label > 0).astype(np.uint8)
        cd_label = Image.fromarray(label)

        if self.opt.phase == "train":
            data = self.transform({"img1": img1, "img2": img2, "cd_label": cd_label})
            img1, img2, cd_label = data["img1"], data["img2"], data["cd_label"]

        img1 = self.normalize(self.to_tensor(img1))
        img2 = self.normalize(self.to_tensor(img2))
        cd_label = torch.from_numpy(np.array(cd_label, dtype=np.int64))

        return {"img1": img1, "img2": img2, "cd_label": cd_label, "fname": fname}


class DataLoader(torch.utils.data.Dataset):
    def __init__(self, opt):
        self.dataset = Load_Dataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=opt.phase == "train",
            pin_memory=True,
            drop_last=opt.phase == "train",
            num_workers=int(opt.num_workers),
        )

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)
