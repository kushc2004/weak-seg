"""Shape/value checks for SEAM training losses and CAM utilities."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from weakseg.weak.losses import (
    adaptive_min_pooling_loss,
    cross_refined_consistency,
    equivariant_loss,
    max_norm,
    max_onehot,
)


def test_max_norm_scales_to_one():
    cam = torch.tensor([[[[0.5, 1.0]], [[-2.0, 4.0]]]])  # (1, 2, 1, 2)
    normalized = max_norm(cam)
    # SEAM divides by (max + eps), so the peak lands just under 1.
    assert abs(normalized[0, 0].max().item() - 1.0) < 1e-3
    assert (normalized >= 0).all()  # ReLU applied
    assert abs(normalized[0, 1, 0, 1].item() - 1.0) < 1e-3


def test_max_onehot_keeps_only_argmax_fg_channel():
    x = torch.zeros(1, 3, 1, 2)  # bg + two fg classes
    x[0, 1, 0, 0] = 0.9
    x[0, 2, 0, 0] = 0.7
    x[0, 1, 0, 1] = 0.2
    x[0, 2, 0, 1] = 0.5
    out = max_onehot(x)
    assert out[0, 1, 0, 0] == 0.9 and out[0, 2, 0, 0] == 0
    assert out[0, 1, 0, 1] == 0 and out[0, 2, 0, 1] == 0.5


def test_adaptive_min_pooling_all_ones_gives_one():
    high = torch.ones(1, 5, 8, 8)
    assert abs(adaptive_min_pooling_loss(high).item() - 1.0) < 1e-6


def test_adaptive_min_pooling_zero_when_flat_zero():
    low = torch.zeros(1, 5, 8, 8)
    assert adaptive_min_pooling_loss(low) == 0.0


def test_adaptive_min_pooling_positive_when_sparse():
    # k = 8*8//4 = 16 pixels; leave only 8 zeros so the bottom quarter mixes in highs.
    mixed = torch.zeros(1, 5, 8, 8)
    mixed[:, :, :7, :] = 10.0
    assert adaptive_min_pooling_loss(mixed).item() > 0


def test_equivariant_loss_small_for_matching_cams():
    torch.manual_seed(3)
    cam = torch.rand(2, 21, 16, 16)
    labels = torch.ones(2, 21, 1, 1)
    small = F.interpolate(cam, scale_factor=0.25, mode="bilinear", align_corners=True)
    loss = equivariant_loss(cam, small, labels)
    # Downscaling the full-res CAM should nearly reproduce the small CAM.
    assert loss.item() < 0.05


def test_cross_refined_consistency_finite_and_reduced():
    other = torch.rand(2, 21, 8, 8)
    refined = torch.rand(2, 21, 8, 8)
    loss = cross_refined_consistency(other, refined, num_classes=21)
    assert torch.isfinite(loss)
    assert 0 <= loss.item() <= 1
