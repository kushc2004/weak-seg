"""CAM extraction and pseudo-mask synthesis checks."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from weakseg.data.labels import load_cls_labels
from weakseg.models.cam_classifier import CamClassifier
from weakseg.weak.cam import extract_cam_scores, generate_cam_scores
from weakseg.weak.crf import CRF_AVAILABLE, dense_crf_inference
from weakseg.weak.pseudo import cams_to_argmax_mask


def test_extract_cam_scores_masks_absent_classes():
    torch.manual_seed(1)
    model = CamClassifier(num_classes=6, pretrained=False)
    label = np.zeros(5, dtype=np.float32)
    label[2] = 1.0  # only fg index 2 present -> VOC class id 3

    view = torch.randn(3, 32, 32)
    scores = extract_cam_scores(model, [(view, False)], torch.from_numpy(label),
                                original_size=(32, 32), device=torch.device("cpu"))
    assert scores.shape == (5, 32, 32)
    for cls_idx in range(5):
        if cls_idx != 2:
            assert scores[cls_idx].sum() == 0
        else:
            assert 0.0 <= scores[cls_idx].max() <= 1.0 + 1e-6


def test_generate_cam_scores_keys_are_voc_ids(synthetic_voc, synthetic_vocab):
    model = CamClassifier(num_classes=len(synthetic_vocab) + 1, pretrained=False)
    ids, labels = load_cls_labels(synthetic_voc, "train", len(synthetic_vocab),
                                  synthetic_vocab)
    outputs = list(generate_cam_scores(model, synthetic_voc, ids[:2], labels[:2],
                                       torch.device("cpu"), scales=(0.5, 1.0)))
    assert len(outputs) == 2
    image_id, cam_dict = outputs[0]
    assert all(1 <= key <= len(synthetic_vocab) for key in cam_dict)
    h, w = next(iter(cam_dict.values())).shape
    assert h > 0 and w > 0


def test_argmax_mask_background_rule():
    h = w = 4
    cam_dict = {3: np.full((h, w), 0.9, dtype=np.float32)}
    mask = cams_to_argmax_mask(cam_dict, bg_alpha=0.26)
    assert mask.shape == (h, w)
    assert (mask == 3).all()

    cam_dict[5] = np.zeros((h, w), dtype=np.float32)  # zero-score class never beats bg
    mask = cams_to_argmax_mask(cam_dict, bg_alpha=0.26)
    assert (mask == 3).all()


def test_crf_graceful_when_unavailable():
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    probs = np.ones((3, 16, 16), dtype=np.float64) / 3
    if CRF_AVAILABLE:
        out = dense_crf_inference(rgb, probs)
        assert out.shape == probs.shape
    else:
        try:
            dense_crf_inference(rgb, probs)
        except RuntimeError as error:
            assert "pydensecrf" in str(error)
        else:
            raise AssertionError("expected RuntimeError when pydensecrf missing")
