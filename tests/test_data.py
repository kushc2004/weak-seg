"""Synthetic VOC layout, label parsing, dataset shapes, palette roundtrip."""
from __future__ import annotations

import numpy as np
import torch

from weakseg.data.datasets import (
    VOCClassificationDataset,
    VOCSegDataset,
    VOCSegInferenceDataset,
    pad_to_multiple,
)
from weakseg.data.labels import filter_labelled, load_cls_labels, read_split_list
from weakseg.utils.voc_palette import colorize_mask, load_class_mask, save_class_mask


def test_synthetic_layout(synthetic_voc):
    for relative in ("JPEGImages", "SegmentationClass", "ImageSets/Segmentation/train.txt",
                     "ImageSets/Segmentation/val.txt", "ImageSets/Main"):
        assert (synthetic_voc / relative).exists(), relative
    train_ids = read_split_list(synthetic_voc / "ImageSets/Segmentation/train.txt")
    val_ids = read_split_list(synthetic_voc / "ImageSets/Segmentation/val.txt")
    assert len(train_ids) + len(val_ids) == 12
    assert not set(train_ids) & set(val_ids)


def test_label_parsing_matches_masks(synthetic_voc, synthetic_vocab):
    ids, labels = load_cls_labels(synthetic_voc, "train", num_fg_classes=len(synthetic_vocab),
                                  class_names=synthetic_vocab)
    assert labels.shape == (len(ids), len(synthetic_vocab))
    # Every positive label must correspond to pixels of that class in the GT mask.
    for index, image_id in enumerate(ids[:4]):
        mask = load_class_mask(synthetic_voc / "SegmentationClass" / f"{image_id}.png")
        present = {int(c) - 1 for c in np.unique(mask) if c > 0}
        predicted = {c for c in range(len(synthetic_vocab)) if labels[index][c] == 1}
        assert present == predicted


def test_filter_labelled_drops_empty_rows():
    ids = ["a", "b", "c"]
    labels = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    kept_ids, kept = filter_labelled(ids, labels)
    assert kept_ids == ["a", "c"]
    assert kept.shape == (2, 2)


def test_classification_dataset_shapes(synthetic_voc, synthetic_vocab):
    ids, labels = load_cls_labels(synthetic_voc, "train", len(synthetic_vocab), synthetic_vocab)
    dataset = VOCClassificationDataset(synthetic_voc, ids, labels, crop_size=48,
                                       resize_range=(48, 64))
    image, target = dataset[0]
    assert image.shape == (3, 48, 48)
    assert target.shape == (len(synthetic_vocab),)
    assert ((image - image.mean()) < 10).all()  # normalized, finite values


def test_seg_dataset_shapes_and_ignore_index(synthetic_voc, synthetic_vocab):
    ids, _ = load_cls_labels(synthetic_voc, "train", len(synthetic_vocab), synthetic_vocab)
    dataset = VOCSegDataset(synthetic_voc, ids, synthetic_voc / "SegmentationClass",
                            crop_size=32, resize_range=(32, 48))
    image, mask = dataset[0]
    assert image.shape == (3, 32, 32)
    assert mask.shape == (32, 32)
    assert mask.dtype == torch.int64


def test_inference_dataset_and_padding(synthetic_voc):
    ids = read_split_list(synthetic_voc / "ImageSets/Segmentation/val.txt")
    dataset = VOCSegInferenceDataset(synthetic_voc, ids)
    image_id, tensor = dataset[0]
    assert isinstance(image_id, str) and tensor.dim() == 3
    padded, original = pad_to_multiple(tensor.unsqueeze(0), multiple=32)
    assert padded.shape[-2] % 32 == 0 and padded.shape[-1] % 32 == 0
    assert original == tensor.shape[1:]


def test_palette_png_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    mask = rng.integers(0, 21, size=(20, 30)).astype(np.uint8)
    path = tmp_path / "m.png"
    save_class_mask(mask, path)
    recovered = load_class_mask(path)
    np.testing.assert_array_equal(mask, recovered)
    assert colorize_mask(mask).size == (30, 20)
