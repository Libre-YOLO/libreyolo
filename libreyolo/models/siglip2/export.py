"""Frozen-class ONNX export for LibreSigLIP2.

Open-vocabulary export (two towers + a tokenizer) is out of scope for v1.
Instead we bake the *current* ``set_classes`` text embeddings into a final
linear projection so the graph is an ordinary ``[B, K]`` image classifier:

    logits = logit_scale.exp() * L2norm(image_tower(x)) @ text_embeds.T + logit_bias

The ``logit_bias`` is included so the exported logits match native for both
softmax (single-label) and sigmoid (multi-label) downstream use. The exported
model is fixed to the labels and input resolution at export time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _FrozenSigLIP2Classifier(nn.Module):
    """Image tower + baked text-embedding linear head -> ``[B, K]`` logits."""

    def __init__(self, vision_model: nn.Module, weight: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        self.vision_model = vision_model
        # weight = logit_scale.exp() * text_embeds, shape [K, D]; bias = logit_bias.
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.vision_model(images)
        feats = F.normalize(feats, dim=-1)
        return feats @ self.weight.t() + self.bias


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
        raise ValueError("LibreSigLIP2 ONNX export needs opset >= 14 (ViT attention).")

    text_embeds = model._text_embeds.detach().to("cpu", torch.float32)
    scale = float(model.model.logit_scale.exp().detach().cpu())
    weight = scale * text_embeds  # [K, D]
    bias = model.model.logit_bias.detach().to("cpu", torch.float32).reshape(())

    original_device = next(model.model.vision_model.parameters()).device
    vision = model.model.vision_model.to("cpu").eval()
    frozen = _FrozenSigLIP2Classifier(vision, weight, bias).eval()

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
                dynamo=False,
            )
    finally:
        model.model.vision_model.to(original_device)
    return output
