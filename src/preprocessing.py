# Image preprocessing pipeline: grayscale, thresholding, binarization, cleaning, resizing

from __future__ import annotations
import numpy as np
from PIL import Image
from typing import Optional, Tuple


def to_grayscale_np(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("L"), dtype=np.uint8)


def otsu_threshold(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    sum_total = np.dot(np.arange(256), hist)

    sum_b = 0.0
    w_b = 0.0
    max_var = -1.0
    threshold = 127

    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break

        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f

        between = w_b * w_f * (m_b - m_f) ** 2
        if between > max_var:
            max_var = between
            threshold = t

    return threshold


def binarize(gray: np.ndarray, invert: bool = False, threshold: Optional[int] = None) -> np.ndarray:
    if threshold is None:
        threshold = otsu_threshold(gray)

    fg = gray <= threshold
    if invert:
        fg = ~fg
    return fg


def crop_to_foreground(mask: np.ndarray, pad: int = 1) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return mask.copy()

    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(mask.shape[0] - 1, y1 + pad)
    x1 = min(mask.shape[1] - 1, x1 + pad)

    return mask[y0:y1 + 1, x0:x1 + 1]


def resize_mask_nn(mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    h, w = mask.shape
    H, W = size

    if h == 0 or w == 0:
        return np.zeros((H, W), dtype=np.float32)

    ys = (np.arange(H) * (h / H)).astype(int)
    xs = (np.arange(W) * (w / W)).astype(int)

    ys = np.clip(ys, 0, h - 1)
    xs = np.clip(xs, 0, w - 1)

    out = mask[ys[:, None], xs[None, :]]
    return out.astype(np.float32)

"""
def remove_small_specks(mask: np.ndarray, min_area: int = 10) -> np.ndarray:
 
    Remove tiny connected components (noise).
 
    comps = connected_components(mask)
 
    keep = np.zeros_like(mask, dtype=bool)
 
    for comp in comps:
 
        if comp.area >= min_area:
 
            keep[comp.pixels[:, 0], comp.pixels[:, 1]] = True
 
    return keep
"""