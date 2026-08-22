"""Hand-computed checks for the confusion-matrix metrics."""
from __future__ import annotations

import numpy as np

from weakseg.evaluation.metrics import ConfusionMatrix


def test_perfect_prediction():
    matrix = ConfusionMatrix(num_classes=3)
    target = np.array([[0, 1], [2, 1]])
    matrix.update(target.copy(), target)
    summary = matrix.summary()
    assert summary["miou"] == 1.0
    assert summary["dice"] == 1.0
    assert summary["pixel_accuracy"] == 1.0
    assert summary["classes_present"] == 3


def test_known_confusion_values():
    # 4 pixels: two class-0 correct, one class-1 predicted as 0, one void ignored.
    pred = np.array([0, 0, 0, 9])
    target = np.array([0, 0, 1, 255])
    matrix = ConfusionMatrix(num_classes=2)
    matrix.update(pred.reshape(2, 2), target.reshape(2, 2))

    tp, fp, fn = matrix.tp_fp_fn
    assert tp.tolist() == [2.0, 0.0]
    assert fp.tolist() == [1.0, 0.0]
    assert fn.tolist() == [0.0, 1.0]

    summary = matrix.summary()
    np.testing.assert_allclose(summary["per_class_iou"], [2 / 3, 0.0], atol=1e-4)
    np.testing.assert_allclose(summary["miou"], (2 / 3 + 0.0) / 2, atol=1e-4)
    np.testing.assert_allclose(matrix.dice()[0], 4 / 5, atol=1e-4)
    np.testing.assert_allclose(summary["pixel_accuracy"], 2 / 3, atol=1e-4)


def test_absent_class_excluded_from_mean():
    matrix = ConfusionMatrix(num_classes=3)
    pred = np.array([1, 1])
    target = np.array([1, 2])
    matrix.update(pred.reshape(1, 2), target.reshape(1, 2))
    summary = matrix.summary()
    # Class 0 absent from GT: mean over classes 1 and 2 only.
    iou = summary["per_class_iou"]
    assert summary["classes_present"] == 2
    np.testing.assert_allclose(summary["miou"], np.mean([iou[1], iou[2]]), atol=1e-4)


def test_shape_mismatch_raises():
    matrix = ConfusionMatrix(num_classes=2)
    try:
        matrix.update(np.zeros((2, 2), dtype=np.uint8), np.zeros((3, 3), dtype=np.uint8))
    except AssertionError:
        return
    raise AssertionError("expected shape mismatch to raise")
