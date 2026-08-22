"""Multi-label classifier whose weights double as a CAM extractor.

The classification head is a 1x1 convolution over the OS16 feature map, so the
global-average-pooled class scores and the class activation maps share weights
exactly (the classic CAM formulation). The naive CAM baseline and SEAM use the
same architecture - SEAM only adds equivariant training losses plus its PCM
refinement branch.
"""
from __future__ import annotations

import torch.nn as nn

from weakseg.models.backbone import ResNetFeatures


class CamClassifier(nn.Module):
    def __init__(self, num_classes: int = 21, pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = ResNetFeatures(pretrained=pretrained)
        self.dropout7 = nn.Dropout2d(0.5)
        self.fc8 = nn.Conv2d(2048, num_classes, kernel_size=1, bias=False)
        nn.init.xavier_uniform_(self.fc8.weight)
        self.from_scratch_modules = [self.fc8]

    def forward(self, x):
        feats = self.backbone(x)
        return self.fc8(self.dropout7(feats["layer4"]))

    def parameter_groups(self, base_lr: float, head_lr_mult: float = 10.0,
                         weight_decay: float = 5e-4) -> list[dict]:
        """Pretrained vs from-scratch groups; heads train at 10x lr (SEAM recipe)."""
        scratch_ids = {id(p) for m in self.from_scratch_modules for p in m.parameters()}
        pretrained_wd, pretrained_bare, scratch_wd, scratch_bare = [], [], [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            is_scratch = id(param) in scratch_ids
            no_decay = name.endswith(".bias") or ".bn" in name
            bucket = (scratch_bare if no_decay else scratch_wd) if is_scratch else (
                pretrained_bare if no_decay else pretrained_wd
            )
            bucket.append(param)

        groups = [
            {"params": pretrained_wd, "lr": base_lr, "weight_decay": weight_decay},
            {"params": pretrained_bare, "lr": base_lr, "weight_decay": 0.0},
            {"params": scratch_wd, "lr": base_lr * head_lr_mult, "weight_decay": weight_decay},
            {"params": scratch_bare, "lr": base_lr * head_lr_mult * 2, "weight_decay": 0.0},
        ]
        return [g for g in groups if g["params"]]
