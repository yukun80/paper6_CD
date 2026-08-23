"""HA-CQI 训练可视化使用的张量辅助函数。"""

import numpy as np
from torchvision import utils


def make_numpy_grid(tensor_data, pad_value=0, padding=0):
    tensor_data = tensor_data.detach()
    vis = utils.make_grid(tensor_data, pad_value=pad_value, padding=padding)
    vis = np.array(vis.cpu()).transpose((1, 2, 0))
    if vis.shape[2] == 1:
        vis = np.stack([vis, vis, vis], axis=-1)
    return vis


def de_norm(tensor_data, mean=None, std=None):
    mean = mean or (0.430, 0.411, 0.296)
    std = std or (0.213, 0.156, 0.143)
    for channel in range(tensor_data.shape[1]):
        tensor_data[:, channel, :, :] = (
            tensor_data[:, channel, :, :] * std[channel] + mean[channel]
        )
    return tensor_data
