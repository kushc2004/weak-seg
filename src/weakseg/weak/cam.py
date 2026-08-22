"""CAM extraction at inference (multi-scale + flip aggregation, SEAM protocol).

Reproduces ``infer_SEAM.py``: sum class maps over scales and horizontal flips,
mask each channel by the image's image-level label, then per-class min-max
normalize (zeroing the floor). For SeamNet the SECOND forward output is used -
the PCM-refined map - exactly as in the original inference script.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F

from weakseg.data.datasets import VOCMultiScaleDataset


@torch.no_grad()
def extract_cam_scores(model, views: list[tuple[torch.Tensor, bool]], label: torch.Tensor,
                       original_size: tuple[int, int], device) -> np.ndarray:
    """Aggregate one image's multi-scale CAM views into normalized fg score maps.

    Returns an ``(num_fg_classes, H, W)`` float32 array; absent classes stay zero.
    """
    model.eval()
    h_orig, w_orig = original_size
    fg_label = label.numpy() if hasattr(label, "numpy") else np.asarray(label)
    accumulator = None

    for tensor, flipped in views:
        cam = model(tensor.unsqueeze(0).to(device))
        if isinstance(cam, tuple):  # SeamNet returns (cam, cam_rv); infer with refined map
            cam = cam[1]
        cam = F.interpolate(cam[:, 1:, :, :], (h_orig, w_orig), mode="bilinear",
                            align_corners=False)[0]
        scores = cam.cpu().numpy() * fg_label[:, None, None]
        if flipped:
            scores = np.flip(scores, axis=-1)
        accumulator = scores if accumulator is None else accumulator + scores

    accumulator[accumulator < 0] = 0
    cam_max = np.max(accumulator, axis=(1, 2), keepdims=True)
    cam_min = np.min(accumulator, axis=(1, 2), keepdims=True)
    accumulator[accumulator < cam_min + 1e-5] = 0
    scores = (accumulator - cam_min - 1e-5) / (cam_max - cam_min + 1e-5)
    scores[fg_label <= 0] = 0.0  # classes absent from this image's labels carry no evidence
    return scores


@torch.no_grad()
def generate_cam_scores(model, voc_root: Path | str, image_ids: list[str],
                        labels: np.ndarray | None, device,
                        scales: tuple[float, ...] = (1.0,), flips: bool = False,
                        max_long_side: int = 960) -> Iterator[tuple[str, dict[int, np.ndarray]]]:
    """Yield ``(image_id, {class_idx: HxW scores})`` for present classes only."""
    dataset = VOCMultiScaleDataset(voc_root, image_ids, labels=labels, scales=scales,
                                   flips=flips, max_long_side=max_long_side)

    for index in range(len(dataset)):
        image_id, views, label, size = dataset[index]
        scores = extract_cam_scores(model, views, label, size, device)
        present = [int(c) for c in np.nonzero(np.asarray(label))[0]]
        yield image_id, {c + 1: scores[c] for c in present}
        if index % 100 == 0:
            print(f"  CAM {index + 1}/{len(dataset)}", flush=True)
