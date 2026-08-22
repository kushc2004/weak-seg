"""Pseudo-mask synthesis from CAM scores: raw thresholding and DenseCRF refinement."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from weakseg.utils.voc_palette import save_class_mask


def cams_to_argmax_mask(cam_dict: dict[int, np.ndarray], bg_alpha: float = 0.26) -> np.ndarray:
    """Argmax over a constant background score plus per-class CAM maps (SEAM default).

    ``cam_dict`` keys are sparse 1-based VOC class ids, so the argmax channel is
    mapped back through ``[0] + sorted(keys)`` to produce full-vocabulary labels.
    """
    if not cam_dict:
        raise ValueError("no CAM scores available for this image")
    classes = sorted(cam_dict)
    shape = next(iter(cam_dict.values())).shape
    fg = np.stack([cam_dict[c] for c in classes])
    bg_score = np.full((1,) + shape, bg_alpha, dtype=np.float32)
    probs = np.concatenate([bg_score, fg], axis=0)
    lookup = np.array([0] + classes, dtype=np.uint8)
    return lookup[np.argmax(probs, axis=0)]


def save_pseudo_mask(mask: np.ndarray, out_dir: Path | str, image_id: str) -> Path:
    path = Path(out_dir) / f"{image_id}.png"
    save_class_mask(mask.astype(np.uint8), path)
    return path


def crf_argmax_mask(image_rgb: np.ndarray, cam_dict: dict[int, np.ndarray],
                    alpha: float = 4.0) -> np.ndarray:
    """DenseCRF-refined argmax mask using SEAM's background rule ``bg=(1-max)^a``.

    Returns class ids in the full VOC vocabulary (0 = background).
    """
    from weakseg.weak.crf import dense_crf_inference  # local import: optional dependency

    classes = sorted(cam_dict)
    fg = np.stack([cam_dict[c] for c in classes])
    bg_score = np.power(1 - np.max(fg, axis=0, keepdims=True), alpha)
    scores = np.concatenate([bg_score, fg], axis=0)
    refined = dense_crf_inference(image_rgb, scores)
    flat = np.argmax(refined, axis=0).astype(np.int64)
    # Remap CRF channel indices back to VOC class ids.
    lookup = np.array([0] + classes, dtype=np.int64)
    return lookup[np.clip(flat, 0, len(classes))].astype(np.uint8)


def crf_probabilities(image_rgb: np.ndarray, cam_dict: dict[int, np.ndarray],
                      alpha: float = 4.0) -> np.ndarray:
    """Background-scored DenseCRF probabilities, index 0 = background."""
    from weakseg.weak.crf import dense_crf_inference

    classes = sorted(cam_dict)
    fg = np.stack([cam_dict[c] for c in classes])
    bg_score = np.power(1 - np.max(fg, axis=0, keepdims=True), alpha)
    scores = np.concatenate([bg_score, fg], axis=0)
    refined = dense_crf_inference(image_rgb, scores)
    return refined
