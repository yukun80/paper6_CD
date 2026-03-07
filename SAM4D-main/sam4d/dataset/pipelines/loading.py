import numpy as np
import torch
from PIL import Image
from mmengine.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadPointsFromFile(object):
    def __init__(self):
        pass

    # @classmethod
    def _load_points(self, pts_filename):
        npz = np.load(pts_filename)
        points = npz[npz.files[0]]
        points = points[:, 0:3]  # only use xyz
        return points

    def __call__(self, results):
        pts_filename = results['pts_filename']
        points = self._load_points(pts_filename)
        # padding one dimension as occupancy channel
        points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
        results['points'] = torch.from_numpy(points)

        return results


@TRANSFORMS.register_module()
class LoadImageFromFile:
    """Load image from a file.
    """

    def __init__(self):
        pass

    def __call__(self, results):
        if "img_filename" not in results:
            return results
        filename = results["img_filename"]
        image = Image.open(filename)
        results["image"] = image
        results["img_ori_shape"] = image.size

        return results
