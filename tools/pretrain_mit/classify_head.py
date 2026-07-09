"""Throwaway ImageNet-1K classification head for pretraining the MiT encoder.

Not part of LibreYOLO's model registry and never imported from anywhere under
``libreyolo/``. It exists only so the scripts in this directory can attach a
classifier on top of the same ``SegformerEncoder`` class that
``LibreSegformer`` uses, train it on ImageNet-1K, then discard the head and
keep only the encoder weights. See ``README.md`` in this directory for the
full rationale and the checkpoint format consumed by
``LibreSegformer(..., pretrained_encoder=...)``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from libreyolo.models.segformer.nn import SIZE_CONFIGS, SegformerEncoder  # noqa: E402


class MiTClassifier(nn.Module):
    """``SegformerEncoder`` + global-average-pool + linear head.

    Inputs are expected already ImageNet-normalized (the classification
    transform pipeline in ``libreyolo/data/classify_dataset.py`` does this),
    unlike ``LibreSegformerNet`` which normalizes raw ``[0, 1]`` tensors
    internally. The encoder itself has no normalization step either way, so
    this has no bearing on later loading the encoder weights back into
    ``LibreSegformerNet``.
    """

    def __init__(self, size: str, num_classes: int = 1000) -> None:
        super().__init__()
        if size not in SIZE_CONFIGS:
            raise ValueError(f"Unknown SegFormer size {size!r}. Must be one of {sorted(SIZE_CONFIGS)}")
        cfg = SIZE_CONFIGS[size]
        self.size = size
        self.encoder = SegformerEncoder(cfg)
        self.head = nn.Linear(cfg.embed_dims[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)[-1]  # last stage feature map, [B, C, H, W]
        pooled = features.mean(dim=(2, 3))
        return self.head(pooled)


__all__ = ["MiTClassifier"]
