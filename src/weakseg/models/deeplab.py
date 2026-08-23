"""DeepLabV3-ResNet50 segmentation model used identically across all experiments."""
from __future__ import annotations

import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights


def build_deeplab(num_classes: int = 21, init: str = "coco") -> nn.Module:
    """Return a DeepLabV3-RN50 with ``num_classes`` outputs.

    ``init="coco"`` loads torchvision's COCO-pretrained VOC weights (used for the
    fully supervised baseline AND all weak runs alike, so every row of the
    comparison shares identical architecture + initialization and only the label
    source differs). Any other value trains from random initialization. For a
    class vocabulary other than 21 the head is re-initialized on top.
    """
    use_coco = init == "coco"
    if use_coco and num_classes != 21:
        raise ValueError("init='coco' provides VOC-21 weights; use init='random' "
                         f"for num_classes={num_classes}")
    weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1 if use_coco else None
    # aux_loss=True keeps the auxiliary head in BOTH builds: COCO weights ship with
    # it, and a fresh build without it would reject those checkpoints at evaluation
    # time (train/eval architecture mismatch).
    model = deeplabv3_resnet50(weights=weights, num_classes=num_classes, aux_loss=True,
                               weights_backbone=None)
    return model


def deeplab_forward(model: nn.Module, images):
    """Convenience accessor returning main logits (ignoring any aux head)."""
    return model(images)["out"]
