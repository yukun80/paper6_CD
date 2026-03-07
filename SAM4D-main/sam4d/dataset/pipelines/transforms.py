import numpy as np
import torch
import torchvision
from mmengine.registry import TRANSFORMS


@TRANSFORMS.register_module()
class ImageResize:
    def __init__(
            self, final_dim,
    ):
        self.final_dim = final_dim

    def __call__(self, input_dict):
        img = input_dict["image"]

        W, H = img.size
        fH, fW = self.final_dim
        resize = (fW / W, fH / H)
        resize_dims = (fW, fH)

        new_img = img.resize(resize_dims)
        transform = torch.eye(4)
        transform[:2, :2] *= torch.tensor(resize, dtype=transform.dtype)

        input_dict["image"] = new_img
        input_dict["img_aug_matrix"] = transform.numpy()
        return input_dict


@TRANSFORMS.register_module()
class ImageNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.compose = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, input_dict):
        input_dict["image"] = self.compose(input_dict["image"])
        input_dict["img_norm_cfg"] = dict(mean=self.mean, std=self.std)
        return input_dict


@TRANSFORMS.register_module()
class SparseVoxelization:
    """Voxel based point sampler.

    Apply voxel sampling to multiple sweep points.

    """

    def __init__(self, voxel_size):
        self.voxel_size = np.array(voxel_size)

    def __call__(self, input_dict):
        points = input_dict["points"]
        coords = (points[:, :3] / self.voxel_size).to(torch.int32)
        coords -= coords.min(0, keepdim=True)[0]

        input_dict['pt_coords'] = coords

        return input_dict
