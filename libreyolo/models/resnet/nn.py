"""ResNet — native LibreYOLO implementation (timm/torchvision-compatible).

A faithful re-implementation of the vanilla ResNet (He et al. 2015, "Deep
Residual Learning for Image Recognition", https://arxiv.org/abs/1512.03385) in
its **v1.5** form: the spatial stride lives on the 3x3 conv of the Bottleneck
(matching timm's ``resnet50`` and torchvision's ``resnet50``), not on the first
1x1. Module and attribute names mirror timm/torchvision exactly so a pretrained
``state_dict`` loads with ``strict=True`` and inference is bit-identical
(enforced by the parity test). See the family NOTICE for attribution.

Implemented sizes: 18/34 (BasicBlock) and 50/101 (Bottleneck). Vanilla stem
(single 7x7 conv) — NOT the ResNet-D deep stem — and no Squeeze-Excite / anti-
alias, matching the ``resnet*.a1_in1k`` (Apache-2.0) checkpoints.
"""

from __future__ import annotations

from typing import Dict, List, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

# size -> (block type, per-stage block counts)
ARCH_DEFS: Dict[str, Dict] = {
    "18": {"block": "basic", "layers": (2, 2, 2, 2)},
    "34": {"block": "basic", "layers": (3, 4, 6, 3)},
    "50": {"block": "bottleneck", "layers": (3, 4, 6, 3)},
    "101": {"block": "bottleneck", "layers": (3, 4, 23, 3)},
}


def conv3x3(in_chs: int, out_chs: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_chs, out_chs, 3, stride=stride, padding=1, bias=False)


def conv1x1(in_chs: int, out_chs: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_chs, out_chs, 1, stride=stride, padding=0, bias=False)


class BasicBlock(nn.Module):
    """ResNet basic block (resnet18/34). Submodules: conv1/bn1, conv2/bn2, downsample."""

    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)  # stride on the first 3x3
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return F.relu(out + identity, inplace=True)


class Bottleneck(nn.Module):
    """ResNet bottleneck (resnet50/101), v1.5: stride on the 3x3 (conv2).

    Submodules: conv1/bn1, conv2/bn2, conv3/bn3, downsample.
    """

    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)  # v1.5: stride here, NOT on conv1
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return F.relu(out + identity, inplace=True)


class ResNet(nn.Module):
    """Vanilla ResNet backbone + classification head (timm-compatible)."""

    def __init__(self, size: str = "50", num_classes: int = 1000):
        super().__init__()
        if size not in ARCH_DEFS:
            raise ValueError(f"Unknown ResNet size {size!r}; choose from {list(ARCH_DEFS)}.")
        spec = ARCH_DEFS[size]
        block: Type[nn.Module] = Bottleneck if spec["block"] == "bottleneck" else BasicBlock
        layers = spec["layers"]
        self.size = size
        self.num_classes = num_classes

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.num_features = 512 * block.expansion
        self.global_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1))
        self.fc = nn.Linear(self.num_features, num_classes)

    def _make_layer(self, block, planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        modules: List[nn.Module] = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            modules.append(block(self.inplanes, planes))
        return nn.Sequential(*modules)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.global_pool(x)
        if pre_logits:
            return x
        return self.fc(x)

    def reset_classifier(self, num_classes: int) -> None:
        """Swap the final Linear for fine-tuning to a new class count."""
        self.num_classes = num_classes
        self.fc = nn.Linear(self.num_features, num_classes).to(self.fc.weight.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_head(self.forward_features(x))
