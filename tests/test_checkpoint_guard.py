"""Guards against salvaged checkpoints saved under the wrong architecture name."""
from __future__ import annotations

import torch

from weakseg.models.cam_classifier import CamClassifier
from weakseg.models.seam import SeamNet
from weakseg.pipeline import CheckpointMismatch, FullPipeline


def _pipeline(tmp_path):
    return FullPipeline(tmp_path, {"fast_dev_run": True, "device": "cpu", "pretrained": False},
                        force=True)


def test_mismatched_checkpoint_is_rejected(tmp_path):
    """A SEAM state dict labelled as 'plain' must be rejected, not force-loaded."""
    pipe = _pipeline(tmp_path)
    seam = SeamNet(num_classes=21, pretrained=False)
    torch.save(seam.state_dict(), pipe.checkpoints_dir / "classifier_plain.pth")
    try:
        pipe._load_classifier_checkpoint("plain")
        raise AssertionError("expected CheckpointMismatch")
    except CheckpointMismatch:
        pass


def test_matching_checkpoint_loads(tmp_path):
    pipe = _pipeline(tmp_path)
    plain = CamClassifier(num_classes=21, pretrained=False)
    torch.save(plain.state_dict(), pipe.checkpoints_dir / "classifier_plain.pth")
    model = pipe._load_classifier_checkpoint("plain")
    assert isinstance(model, CamClassifier)
