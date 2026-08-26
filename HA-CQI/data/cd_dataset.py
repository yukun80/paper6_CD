from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .runtime_snapshot import scan_split_files
from .transform import Transforms
from .tif_io import is_tiff_path, read_binary_label_tif, read_sar_tif


def seed_worker(worker_id: int) -> None:
    """把 PyTorch worker seed 同步给 Python 与 NumPy 增强随机源。"""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _normalize_to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    if image.mode in {"L", "I", "F", "P", "LA"}:
        return image.convert("RGB")
    return image.convert("RGB")


def _load_image(path: Path):
    """兼容普通图像与单波段 SAR tif。"""
    if is_tiff_path(path):
        return read_sar_tif(path)
    return _normalize_to_rgb(Image.open(path))


def _load_label(path: Path) -> Image.Image:
    if is_tiff_path(path):
        return Image.fromarray(read_binary_label_tif(path))
    label = np.array(Image.open(path).convert("L"), dtype=np.uint8)
    label = (label > 0).astype(np.uint8)
    return Image.fromarray(label)


def _to_tensor_image(image, to_tensor: transforms.ToTensor) -> torch.Tensor:
    if torch.is_tensor(image):
        return image.float()
    return to_tensor(image)


def _to_label_tensor(label) -> torch.Tensor:
    if torch.is_tensor(label):
        if label.ndim == 3 and label.shape[0] == 1:
            label = label.squeeze(0)
        return label.to(dtype=torch.int64)
    return torch.from_numpy(np.array(label, dtype=np.int64))


class Load_Dataset(Dataset):
    """读取标准变化检测目录，并兼容 SAR tif / PNG 输入。"""

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        # train/val loader 共用同一 argparse Namespace；冻结 phase，避免后续切换污染验证增强。
        self.phase = str(opt.phase)

        dataset_root = Path(opt.dataroot) / opt.dataset
        split_files = scan_split_files(dataset_root, self.phase)
        self.t1_map = split_files.a
        self.t2_map = split_files.b
        self.label_map = split_files.label
        self.fnames = list(split_files.filenames)

        self.dataset_size = len(self.fnames)
        self.normalize = transforms.Normalize(tuple(opt.mean), tuple(opt.std))
        self.transform = Transforms(
            input_size=opt.input_size,
            dataset_mode=opt.dataset_mode,
            radiometric_jitter_mode=str(
                getattr(opt, "radiometric_jitter_mode", "shared")
            ),
        )
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        fname = self.fnames[index]
        img1 = _load_image(self.t1_map[fname])
        img2 = _load_image(self.t2_map[fname])
        cd_label = _load_label(self.label_map[fname])

        if self.phase == "train":
            data = self.transform({"img1": img1, "img2": img2, "cd_label": cd_label})
            img1, img2, cd_label = data["img1"], data["img2"], data["cd_label"]

        img1 = self.normalize(_to_tensor_image(img1, self.to_tensor))
        img2 = self.normalize(_to_tensor_image(img2, self.to_tensor))
        cd_label = _to_label_tensor(cd_label)

        return {"img1": img1, "img2": img2, "cd_label": cd_label, "fname": fname}


class DataLoader(torch.utils.data.Dataset):
    def __init__(self, opt):
        self.dataset = Load_Dataset(opt)
        self.generator = torch.Generator()
        phase_offset = 0 if opt.phase == "train" else 100_000
        self.generator.manual_seed(int(getattr(opt, "seed", 1)) + phase_offset)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=opt.phase == "train",
            pin_memory=True,
            drop_last=opt.phase == "train",
            num_workers=int(opt.num_workers),
            persistent_workers=False,
            worker_init_fn=seed_worker,
            generator=self.generator,
        )

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)

    def get_generator_state(self) -> torch.Tensor:
        return self.generator.get_state()

    def set_generator_state(self, state: torch.Tensor) -> None:
        self.generator.set_state(state)
