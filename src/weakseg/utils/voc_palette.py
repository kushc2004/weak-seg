"""PASCAL VOC color palette and mask PNG I/O.

VOC stores segmentation masks as palettized PNGs where pixel value == class id
(0..20 foreground+background, 255 void). We reproduce that encoding so exported
pseudo-masks are directly comparable with ``SegmentationClass`` ground truth.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def voc_palette(num_classes: int = 256) -> list[tuple[int, int, int]]:
    """Standard PASCAL VOC colormap as a list of RGB tuples."""
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for i in range(num_classes):
        lab = i
        r = g = b = 0
        for j in range(8):
            r |= ((lab >> 0) & 1) << (7 - j)
            g |= ((lab >> 1) & 1) << (7 - j)
            b |= ((lab >> 2) & 1) << (7 - j)
            lab >>= 3
        palette[i] = (r, g, b)
    return [tuple(int(v) for v in c) for c in palette]


def save_class_mask(mask: np.ndarray, path) -> None:
    """Save an HxW uint8 class-id array as a VOC-style palettized PNG."""
    assert mask.ndim == 2 and mask.dtype == np.uint8, f"bad mask {mask.shape} {mask.dtype}"
    pal = voc_palette()
    flat = b""
    for r, g, b in pal:
        flat += bytes((r, g, b))
    img = Image.fromarray(mask, mode="P")
    img.putpalette([v for rgb in pal for v in rgb])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def load_class_mask(path) -> np.ndarray:
    """Load a VOC-style palettized PNG as an HxW uint8 class-id array."""
    img = Image.open(path)
    return np.asarray(img, dtype=np.uint8)


def colorize_mask(mask: np.ndarray) -> Image.Image:
    """Render a class-id mask as an RGB PIL image using the VOC palette."""
    pal = np.array(voc_palette(), dtype=np.uint8)
    rgb = pal[np.clip(mask.astype(np.int64), 0, 255)]
    return Image.fromarray(rgb, mode="RGB")
