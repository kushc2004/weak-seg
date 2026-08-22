from weakseg.weak.losses import (
    adaptive_min_pooling_loss,
    max_norm,
    max_onehot,
)
from weakseg.weak.cam import generate_cam_scores
from weakseg.weak.pseudo import cams_to_argmax_mask, save_pseudo_mask

__all__ = [
    "adaptive_min_pooling_loss",
    "max_norm",
    "max_onehot",
    "generate_cam_scores",
    "cams_to_argmax_mask",
    "save_pseudo_mask",
]
