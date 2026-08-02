"""Native VGG-16/VGG-19 image-classification networks.

The graph and module names follow torchvision's BSD-3-Clause VGG implementation
at commit ``10f68dbd78b9aa5cab9328f3b2e99cfb0b608122`` so official state dicts
load strictly and can be compared tensor-for-tensor. See the family NOTICE for
the complete provenance record.
"""

from __future__ import annotations

from typing import Dict, List, Union

import torch
import torch.nn as nn

LayerSpec = Union[int, str]

CONFIGS: Dict[str, List[LayerSpec]] = {
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}

ARCH_DEFS = {
    "16": ("D", False),
    "19": ("E", False),
    "16bn": ("D", True),
    "19bn": ("E", True),
}


def make_layers(config: List[LayerSpec], batch_norm: bool = False) -> nn.Sequential:
    """Build a VGG feature stack while retaining upstream module indices."""
    layers: List[nn.Module] = []
    in_channels = 3
    for value in config:
        if value == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            continue
        out_channels = int(value)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        if batch_norm:
            layers.extend((conv, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)))
        else:
            layers.extend((conv, nn.ReLU(inplace=True)))
        in_channels = out_channels
    return nn.Sequential(*layers)


class VGG(nn.Module):
    """Configuration D/E VGG classifier, with optional batch normalization."""

    def __init__(
        self,
        size: str = "16",
        num_classes: int = 1000,
        dropout: float = 0.5,
        init_weights: bool = True,
    ) -> None:
        super().__init__()
        if size not in ARCH_DEFS:
            raise ValueError(
                f"Unknown VGG size {size!r}; choose from {list(ARCH_DEFS)}."
            )
        config_name, batch_norm = ARCH_DEFS[size]
        self.size = size
        self.num_classes = num_classes
        self.features = make_layers(CONFIGS[config_name], batch_norm=batch_norm)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

    def reset_classifier(self, num_classes: int) -> None:
        """Replace the final linear layer while preserving device and dtype."""
        self.num_classes = num_classes
        old = self.classifier[6]
        assert isinstance(old, nn.Linear)
        self.classifier[6] = nn.Linear(4096, num_classes).to(
            device=old.weight.device,
            dtype=old.weight.dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


__all__ = ["ARCH_DEFS", "CONFIGS", "VGG", "make_layers"]
