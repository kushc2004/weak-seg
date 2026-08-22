"""Segmentation metrics: mIoU, Dice, pixel accuracy from a running confusion matrix.

Label 255 (VOC void) is ignored everywhere. mIoU/Dice average over classes that
appear in the evaluated ground truth, matching the standard VOC protocol.
"""
from __future__ import annotations

import numpy as np


class ConfusionMatrix:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        """Accumulate one pair of HxW arrays (pred/target class ids, void=255)."""
        assert pred.shape == target.shape, f"shape mismatch {pred.shape} vs {target.shape}"
        valid = target != 255
        if not valid.any():
            return
        p = pred[valid].astype(np.int64)
        t = target[valid].astype(np.int64)
        keep = (p >= 0) & (p < self.num_classes)
        np.add.at(self.matrix, (t[keep], p[keep]), 1)

    @property
    def tp_fp_fn(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tp = np.diag(self.matrix).astype(np.float64)
        fp = self.matrix.sum(axis=0) - tp
        fn = self.matrix.sum(axis=1) - tp
        return tp, fp, fn

    def iou(self) -> np.ndarray:
        tp, fp, fn = self.tp_fp_fn
        denom = tp + fp + fn
        return np.divide(tp, denom, out=np.zeros_like(tp), where=denom > 0)

    def dice(self) -> np.ndarray:
        tp, fp, fn = self.tp_fp_fn
        denom = 2 * tp + fp + fn
        return np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)

    def pixel_accuracy(self) -> float:
        total = self.matrix.sum()
        return float(np.diag(self.matrix).sum() / total) if total else 0.0

    def summary(self) -> dict:
        iou, dice = self.iou(), self.dice()
        present = self.matrix.sum(axis=1) > 0
        return {
            "miou": round(float(iou[present].mean()) if present.any() else 0.0, 5),
            "dice": round(float(dice[present].mean()) if present.any() else 0.0, 5),
            "miou_all_classes": round(float(iou.mean()), 5),
            "pixel_accuracy": round(self.pixel_accuracy(), 5),
            "per_class_iou": [round(float(v), 5) for v in iou],
            "classes_present": int(present.sum()),
        }
