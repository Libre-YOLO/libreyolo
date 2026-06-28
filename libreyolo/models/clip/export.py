"""Frozen-class ONNX export for LibreCLIP.

Open-vocabulary export (two towers + a tokenizer) is awkward and out of scope
for v1. Instead we bake the *current* ``set_classes`` text embeddings into a
final linear projection so the graph is an ordinary ``[B, K]`` image classifier:

    logits = (logit_scale.exp() * text_embeds) @ L2norm(image_tower(x))

The exported model is fixed to the labels and input resolution at export time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _FrozenCLIPClassifier(nn.Module):
    """Image tower + baked text-embedding linear head → ``[B, K]`` logits."""

    def __init__(self, visual: nn.Module, weight: torch.Tensor):
        super().__init__()
        self.visual = visual
        # weight = logit_scale.exp() * text_embeds, shape [K, D]
        self.register_buffer("weight", weight)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.visual(images)
        feats = F.normalize(feats, dim=-1)
        return feats @ self.weight.t()


def export_frozen_onnx(
    model,
    imgsz: Optional[int] = None,
    opset: int = 14,
    output: Optional[str] = None,
    dynamic_batch: bool = True,
) -> str:
    """Export ``model`` (with its current classes) to a frozen-class ONNX file."""
    res = int(imgsz or model.input_size)
    if opset < 14:
        raise ValueError("LibreCLIP ONNX export needs opset >= 14 (ViT attention).")

    text_embeds = model._text_embeds.detach().to("cpu", torch.float32)
    scale = float(model.model.logit_scale.exp().detach().cpu())
    weight = scale * text_embeds  # [K, D]

    original_device = next(model.model.visual.parameters()).device
    visual = model.model.visual.to("cpu").eval()
    frozen = _FrozenCLIPClassifier(visual, weight).eval()

    if output is None:
        output = f"{model.FILENAME_PREFIX}{model.size}-cls.onnx"
    output = str(output)
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.zeros(1, 3, res, res, dtype=torch.float32)
    dynamic_axes = (
        {"images": {0: "batch"}, "logits": {0: "batch"}} if dynamic_batch else None
    )
    try:
        with torch.no_grad():
            torch.onnx.export(
                frozen,
                dummy,
                output,
                input_names=["images"],
                output_names=["logits"],
                opset_version=opset,
                dynamic_axes=dynamic_axes,
                # Stable TorchScript exporter (no onnxscript dependency); the
                # frozen graph is a plain ViT + matmul that exports cleanly.
                dynamo=False,
            )
    finally:
        # Restore the tower to its original device even if export raised, so the
        # model instance stays usable for further predict()/_forward() calls.
        model.model.visual.to(original_device)
    return output
