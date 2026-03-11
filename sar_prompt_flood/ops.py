"""图像/二值形态学兼容操作。"""

from __future__ import annotations

from collections import deque
from typing import List, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None


def binary_open(mask: np.ndarray, ksize: int = 3) -> np.ndarray:
    if cv2 is not None:
        return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((ksize, ksize), np.uint8))
    return binary_dilate(binary_erode(mask, ksize), ksize)


def binary_close(mask: np.ndarray, ksize: int = 3) -> np.ndarray:
    if cv2 is not None:
        return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((ksize, ksize), np.uint8))
    return binary_erode(binary_dilate(mask, ksize), ksize)


def binary_dilate(mask: np.ndarray, ksize: int = 3, iterations: int = 1) -> np.ndarray:
    out = mask.astype(np.uint8)
    if cv2 is not None:
        return cv2.dilate(out, np.ones((ksize, ksize), np.uint8), iterations=iterations)
    for _ in range(int(iterations)):
        out = _binary_max_filter(out, ksize)
    return out


def binary_erode(mask: np.ndarray, ksize: int = 3) -> np.ndarray:
    out = mask.astype(np.uint8)
    if cv2 is not None:
        return cv2.erode(out, np.ones((ksize, ksize), np.uint8), iterations=1)
    pad = ksize // 2
    padded = np.pad(out, ((pad, pad), (pad, pad)), mode="constant", constant_values=0)
    result = np.zeros_like(out)
    for y in range(out.shape[0]):
        for x in range(out.shape[1]):
            window = padded[y : y + ksize, x : x + ksize]
            result[y, x] = 1 if np.all(window > 0) else 0
    return result


def connected_components_with_stats(mask: np.ndarray) -> Tuple[int, np.ndarray, List[dict]]:
    mask = mask.astype(np.uint8)
    if cv2 is not None:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        stat_list = []
        for i in range(num_labels):
            stat_list.append({"area": int(stats[i, cv2.CC_STAT_AREA])})
        return int(num_labels), labels, stat_list

    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    stats: List[dict] = [{"area": 0}]
    label_id = 0
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for y in range(h):
        for x in range(w):
            if mask[y, x] == 0 or labels[y, x] != 0:
                continue
            label_id += 1
            q = deque([(y, x)])
            labels[y, x] = label_id
            area = 0
            while q:
                cy, cx = q.popleft()
                area += 1
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] > 0 and labels[ny, nx] == 0:
                        labels[ny, nx] = label_id
                        q.append((ny, nx))
            stats.append({"area": area})
    return label_id + 1, labels, stats


def draw_disk(mask: np.ndarray, center: Tuple[int, int], radius: int, value: int) -> np.ndarray:
    out = mask.copy()
    x0, y0 = center
    yy, xx = np.ogrid[: mask.shape[0], : mask.shape[1]]
    disk = (xx - int(x0)) ** 2 + (yy - int(y0)) ** 2 <= int(radius) ** 2
    out[disk] = value
    return out


def gradient_magnitude(arr: np.ndarray) -> np.ndarray:
    if cv2 is not None:
        grad_x = cv2.Sobel(arr.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(arr.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    else:
        grad_y, grad_x = np.gradient(arr.astype(np.float32))
    return np.abs(grad_x) + np.abs(grad_y)


def _binary_max_filter(mask: np.ndarray, ksize: int) -> np.ndarray:
    pad = ksize // 2
    padded = np.pad(mask, ((pad, pad), (pad, pad)), mode="constant", constant_values=0)
    result = np.zeros_like(mask)
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            window = padded[y : y + ksize, x : x + ksize]
            result[y, x] = 1 if np.any(window > 0) else 0
    return result
