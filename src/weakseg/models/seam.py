"""SEAM network: equivariant CAM training + Pixel Correlation Module refinement.

Ported from YudeWang/SEAM (CVPR 2020, MIT license), ``network/resnet38_SEAM.py``,
adapted to the torchvision ResNet-50 backbone. Two differences from the original:

* the backbone is ImageNet-pretrained ResNet-50 with OS16 layer4 (see backbone.py);
* the f8_3 branch taps stride-8 features and is bilinearly resized down to the CAM
  grid (the Wide ResNet-38 original had all branches at stride 8).

The Pixel Correlation Module propagates the (detached, normalized) CAM across an
affinity matrix computed from learned projections of mid-level features plus the
image itself, expanding activations beyond discriminative object parts.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from weakseg.models.cam_classifier import CamClassifier


class SeamNet(CamClassifier):
    def __init__(self, num_classes: int = 21, pretrained: bool = True):
        super().__init__(num_classes=num_classes, pretrained=pretrained)
        self.f8_3 = nn.Conv2d(512, 64, kernel_size=1, bias=False)    # <- layer2 (512ch)
        self.f8_4 = nn.Conv2d(1024, 128, kernel_size=1, bias=False)  # <- layer3 (1024ch)
        self.f9 = nn.Conv2d(192 + 3, 192, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.f8_3.weight)
        nn.init.kaiming_normal_(self.f8_4.weight)
        nn.init.xavier_uniform_(self.f9.weight, gain=4)
        self.from_scratch_modules = [self.fc8, self.f8_3, self.f8_4, self.f9]

    @staticmethod
    def _normalize_cam(cam: torch.Tensor) -> torch.Tensor:
        """Background-augmented one-hot-suppressed CAM (verbatim SEAM semantics)."""
        n, c, _, _ = cam.size()
        with torch.no_grad():
            cam_d = F.relu(cam.detach())
            cam_max = torch.max(cam_d.view(n, c, -1), dim=-1)[0].view(n, c, 1, 1) + 1e-5
            norm = F.relu(cam_d - 1e-5) / cam_max
            norm[:, 0, :, :] = 1 - torch.max(norm[:, 1:, :, :], dim=1)[0]
            fg_max = torch.max(norm[:, 1:, :, :], dim=1, keepdim=True)[0]
            norm[:, 1:, :, :][norm[:, 1:, :, :] < fg_max] = 0
        return norm

    def pcm(self, cam_norm: torch.Tensor, image: torch.Tensor,
            layer2: torch.Tensor, layer3: torch.Tensor) -> torch.Tensor:
        """Affinity propagation of the normalized CAM over pixel correlations."""
        h, w = cam_norm.shape[-2:]
        cam_flat = cam_norm.flatten(2)                                   # (n, C, hw)

        low = F.interpolate(layer2.detach(), size=(h, w), mode="bilinear", align_corners=True)
        high = F.interpolate(layer3.detach(), size=(h, w), mode="bilinear", align_corners=True)
        feat = torch.cat([
            F.interpolate(image, size=(h, w), mode="bilinear", align_corners=True),
            F.relu(self.f8_3(low)),
            F.relu(self.f8_4(high)),
        ], dim=1)
        feat = self.f9(feat).flatten(2)                                  # (n, 192, hw)
        feat = feat / (torch.norm(feat, dim=1, keepdim=True) + 1e-5)

        aff = F.relu(torch.matmul(feat.transpose(1, 2), feat))
        aff = aff / (torch.sum(aff, dim=1, keepdim=True) + 1e-5)
        return torch.matmul(cam_flat, aff).view(-1, cam_norm.size(1), h, w)

    def forward(self, x: torch.Tensor):
        """Return ``(cam, cam_rv)`` upsampled to input resolution.

        At inference SEAM uses the SECOND output (``infer_SEAM.py`` reads
        ``_, cam = model(img)``): the PCM-refined map.
        """
        feats = self.backbone(x)
        n, _, h, w = x.shape
        cam = self.fc8(self.dropout7(feats["layer4"]))
        cam_norm = self._normalize_cam(cam)
        cam_rv = self.pcm(cam_norm, x, feats["layer2"], feats["layer3"])
        return (
            F.interpolate(cam, (h, w), mode="bilinear", align_corners=True),
            F.interpolate(cam_rv, (h, w), mode="bilinear", align_corners=True),
        )
