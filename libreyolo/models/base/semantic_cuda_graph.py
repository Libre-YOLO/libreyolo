"""CUDA-graph training capture for the dense-logits semantic families.

SegFormer and LingBot-Vision train the same way: encode the image, predict
per-class logits at the network's native (patch or stride-4) resolution,
upsample to the label grid, then cross-entropy with an ignore index. Only
the first half can be captured — the upsample target size comes from the
labels, and the all-pixels-ignored guard is a host sync.

A family opts in by exposing that split on its network module:

``forward_logits(images) -> logits``
    Pure function of the input at a fixed input shape.
``loss_from_logits(logits, targets) -> {"total_loss": ..., ...}``
    The eager remainder, called by the family's own ``forward`` too, so the
    graphed and eager paths cannot drift.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["SemanticLogitsCudaGraphMixin"]


class _LogitsOnly(nn.Module):
    """Adapter exposing ``forward_logits`` as a plain forward for capture."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        return self.model.forward_logits(imgs)


class SemanticLogitsCudaGraphMixin:
    """Provide ``cuda_graph_train_spec`` for dense-logits semantic training.

    Mixed in ahead of ``BaseTrainer`` so it overrides the base hook. Returns
    ``None`` — plain eager training — for any other task, or for a network
    that does not expose the two-method split.
    """

    def cuda_graph_train_spec(self):
        from libreyolo.training.cuda_graph import (
            CudaGraphTrainSpec,
            GraphableNetwork,
        )

        task = getattr(getattr(self, "wrapper_model", None), "task", None)
        if task != "semantic":
            return None
        model = getattr(self, "model", None)
        raw = getattr(model, "module", model)
        if not isinstance(raw, nn.Module):
            return None
        if not (hasattr(raw, "forward_logits") and hasattr(raw, "loss_from_logits")):
            return None

        network = GraphableNetwork(_LogitsOnly(raw))

        def assemble(flat, imgs, targets, polygons=None):
            return raw.loss_from_logits(network.rebuild(flat), targets)

        return CudaGraphTrainSpec(network=network, assemble=assemble)
