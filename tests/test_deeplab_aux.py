"""Train/eval architecture consistency for DeepLabV3 checkpoints.

Regression: checkpoints saved from a COCO-initialized model (which includes the
aux classifier) must load into a freshly built evaluation model.
"""
from __future__ import annotations

import torch

from weakseg.models.deeplab import build_deeplab


def test_checkpoint_saved_with_coco_arch_loads_into_fresh_build(tmp_path):
    trained = build_deeplab(num_classes=21, init="random")  # same module layout as coco build
    torch.save(trained.state_dict(), tmp_path / "seg.pth")

    evaluator = build_deeplab(num_classes=21, init="random")
    evaluator.load_state_dict(torch.load(tmp_path / "seg.pth", map_location="cpu"))


def test_aux_head_present_regardless_of_init():
    for init in ("coco", "random"):
        model = build_deeplab(num_classes=21, init=init)
        assert model.aux_classifier is not None, f"aux head missing for init={init}"
