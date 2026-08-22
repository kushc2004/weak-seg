"""Model forward-shape and parameter-group checks (tiny inputs, CPU)."""
from __future__ import annotations

import pytest
import torch

from weakseg.models.cam_classifier import CamClassifier
from weakseg.models.deeplab import build_deeplab
from weakseg.models.seam import SeamNet


@pytest.fixture(scope="module")
def tiny_image():
    torch.manual_seed(0)
    return torch.randn(2, 3, 64, 64)


def test_cam_classifier_forward(tiny_image):
    model = CamClassifier(num_classes=6, pretrained=False)
    cam = model(tiny_image)
    assert cam.shape == (2, 6, 4, 4)  # stride 16


def test_seam_forward_returns_refined_pair(tiny_image):
    model = SeamNet(num_classes=6, pretrained=False)
    cam, refined = model(tiny_image)
    assert cam.shape == (2, 6, 64, 64)
    assert refined.shape == (2, 6, 64, 64)
    assert torch.isfinite(cam).all() and torch.isfinite(refined).all()


def test_seam_backward_flows_to_pcm_branch():
    model = SeamNet(num_classes=6, pretrained=False)
    image = torch.randn(1, 3, 64, 64)
    _, refined = model(image)
    refined.sum().backward()
    assert model.f9.weight.grad is not None          # PCM projections receive gradients
    assert model.fc8.weight.grad is None             # CAM head is detached from this path


def test_parameter_groups_split_pretrained_and_scratch():
    model = CamClassifier(num_classes=21, pretrained=False)
    groups = model.parameter_groups(base_lr=0.01)
    scratch = [p for g in groups[2:] for p in g["params"]]
    pretrained = [p for g in groups[:2] for p in g["params"]]
    assert any(p is model.fc8.weight for p in scratch)
    assert len(pretrained) > 50
    assert all(g["lr"] >= 0.01 for g in groups)


def test_deeplab_builder_random_init_small_vocab(tiny_image):
    model = build_deeplab(num_classes=6, init="random").eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 64, 64))["out"]
    assert out.shape == (1, 6, 64, 64)


def test_deeplab_coco_requires_21_classes():
    # COCO weights exist only for the 21-class VOC vocabulary; other sizes must raise.
    with pytest.raises(ValueError):
        build_deeplab(num_classes=6, init="coco")
