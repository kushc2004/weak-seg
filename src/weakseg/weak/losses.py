"""SEAM training losses (ported verbatim-in-spirit from YudeWang/SEAM train_SEAM.py).

Loss composition per step (both full-res and 0.3x-downscaled views):

    loss = multilabel_soft_margin(GAP(cam), y)          # image-level supervision
         + adaptive_min_pooling(cam_rv * y)             # expand foreground coverage
         + L1(downscale(max_norm(cam_full)), cam_small) # scale-equivariance (loss_er)
         + top-k |max_onehot(cam_other) - cam_rv_self|  # cross-view consistency (loss_ecr)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def adaptive_min_pooling_loss(x: torch.Tensor) -> torch.Tensor:
    """Push up the weakest quarter of foreground activations (SEAM's alpha-shaper)."""
    n, _, h, w = x.size()
    k = max(1, h * w // 4)
    x = torch.max(x, dim=1)[0]
    values = torch.topk(x.reshape(n, -1), k=k, dim=-1, largest=False)[0]
    return torch.sum(F.relu(values)) / (k * n)


def max_norm(pred: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Per-channel spatial max normalization of (optionally ReLU'd) CAMs to [0, 1]."""
    n, c, h, w = pred.size()
    flat = F.relu(pred).reshape(n, c, -1)
    flat = flat / (flat.max(dim=-1, keepdim=True)[0] + eps)
    return flat.view(n, c, h, w)


def max_onehot(x: torch.Tensor) -> torch.Tensor:
    """Keep only the argmax foreground channel per pixel (background channel untouched)."""
    n, c, h, w = x.size()
    fg_max = torch.max(x[:, 1:, :, :], dim=1, keepdim=True)[0]
    suppressed = x.clone()
    suppressed[:, 1:, :, :][x[:, 1:, :, :] != fg_max] = 0
    return suppressed


def equivariant_loss(cam_full: torch.Tensor, cam_small: torch.Tensor,
                     labels: torch.Tensor, scale_factor: float = 0.3) -> torch.Tensor:
    """``loss_er``: CAMs must be equivariant to image rescaling.

    ``cam_full`` and ``cam_small`` come from the full-res and downscaled views;
    the full-res CAM is max-normalized, label-masked, downsampled to the small
    view's grid, then compared with plain L1 over foreground channels.
    """
    small_size = cam_small.shape[-2:]
    cam_full_ds = F.interpolate(
        max_norm(cam_full) * labels, size=small_size, mode="bilinear", align_corners=True
    )
    cam_small_n = F.interpolate(
        max_norm(cam_small), size=small_size, mode="bilinear", align_corners=True
    ) * labels
    return torch.mean(torch.abs(cam_full_ds[:, 1:, :, :] - cam_small_n[:, 1:, :, :]))


def cross_refined_consistency(cam_other: torch.Tensor, cam_rv_self: torch.Tensor,
                              num_classes: int, top_fraction: float = 0.2) -> torch.Tensor:
    """``loss_ecr``: PCM-refined CAM must agree with the other view's raw argmax.

    Only the hardest ``top_fraction`` of pixels contribute (SEAM's top-k mean),
    which focuses learning on object boundaries and missed regions.
    """
    n, c, h, w = cam_rv_self.shape
    residual = torch.abs(max_onehot(cam_other.detach()) - cam_rv_self)
    k = int(num_classes * h * w * top_fraction)
    return torch.mean(torch.topk(residual.reshape(n, -1), k=k, dim=-1)[0])
