"""Image-level label handling.

Weak-supervision training reads ONLY the official ``ImageSets/Main`` classification
annotations - the pixel masks in ``SegmentationClass`` are reserved for evaluation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from weakseg import VOC_NUM_FG_CLASSES


def read_split_list(path: Path | str) -> list[str]:
    """Read a VOC split file (one image id per line)."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def load_cls_labels(voc_root: Path | str, split: str, num_fg_classes: int = VOC_NUM_FG_CLASSES,
                    class_names: list[str] | None = None) -> tuple[list[str], np.ndarray]:
    """Return ``(image_ids, label_matrix)`` from ``ImageSets/Main/<cls>_<split>.txt``.

    The matrix is ``(n, num_fg_classes)`` float32 with 1.0 for present classes.
    Images whose official Main annotation files mark them as hard ("0") or absent
    ("-1") receive a 0 for that class, matching the honest image-level protocol.
    """
    voc_root = Path(voc_root)
    if class_names is None:
        class_names_path = voc_root / "classes.txt"
        if class_names_path.is_file():
            names = [line.strip() for line in class_names_path.read_text().splitlines() if line.strip()]
            class_names = names[1:]
        else:
            from weakseg import VOC_CLASS_NAMES

            class_names = VOC_CLASS_NAMES[1:]
    assert len(class_names) == num_fg_classes, (len(class_names), num_fg_classes)

    per_image: dict[str, np.ndarray] = {}
    for cls_idx, name in enumerate(class_names):
        label_file = voc_root / "ImageSets" / "Main" / f"{name}_{split}.txt"
        if not label_file.is_file():
            raise FileNotFoundError(
                f"Missing image-level labels: {label_file}. VOC2012 ships these in ImageSets/Main."
            )
        for line in label_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            image_id, raw = line.rsplit(None, 1)
            flag = int(raw)
            if image_id not in per_image:
                per_image[image_id] = np.zeros(num_fg_classes, dtype=np.float32)
            if flag == 1:
                per_image[image_id][cls_idx] = 1.0

    ids = sorted(per_image)
    matrix = np.stack([per_image[image_id] for image_id in ids])
    return ids, matrix


def filter_labelled(ids: list[str], labels: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Drop images with no positive class (nothing to learn from in multi-label BCE)."""
    keep = labels.sum(axis=1) > 0
    return [i for i, k in zip(ids, keep) if k], labels[keep]
