"""ResNet-50 backbone with output-stride 16, exposing intermediate features.

The original SEAM implementation builds on an MXNet-trained Wide ResNet-38
(PyTorch 0.4 era). We port the SEAM heads onto torchvision's ImageNet-
pretrained ResNet-50 instead - the channel mapping is exact:

    ResNet-38 conv4 (512ch)  ->  ResNet-50 layer2 (512ch, stride 8)
    ResNet-38 conv5 (1024ch) ->  ResNet-50 layer3 (1024ch, stride 16)
    ResNet-38 conv6 (4096ch) ->  ResNet-50 layer4 (2048ch, stride 16)

Layer 4 keeps stride 16 via dilation (``replace_stride_with_dilation``),
which preserves spatial resolution for CAMs.
"""
from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet50
from torchvision.models.resnet import ResNet50_Weights


class ResNetFeatures(nn.Module):
    """Frozen-layout ResNet-50 returning the three feature maps SEAM consumes."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnet50(weights=weights, replace_stride_with_dilation=[False, False, True])
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = (
            net.layer1, net.layer2, net.layer3, net.layer4,
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        layer2 = self.layer2(x)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return {"layer2": layer2, "layer3": layer3, "layer4": layer4}

    def pretrained_parameters(self):
        for module in (self.stem, self.layer1, self.layer2, self.layer3, self.layer4):
            yield from module.parameters()
