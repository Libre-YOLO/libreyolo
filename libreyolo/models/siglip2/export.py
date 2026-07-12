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

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.serialization import SCHEMA_VERSION


class _FrozenSigLIP2Classifier(nn.Module):
    """Image tower + baked text-embedding linear head -> ``[B, K]`` logits."""

    def __init__(
        self, vision_model: nn.Module, weight: torch.Tensor, bias: torch.Tensor
    ):
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
    opset: Optional[int] = None,
    output: Optional[str] = None,
    *,
    batch: int = 1,
    dynamic: bool = True,
    device: str | torch.device | int | None = None,
    simplify: bool = True,
    verbose: bool = False,
) -> str:
    """Export ``model`` (with its current classes) to a frozen-class ONNX file."""
    from ...export.onnx import _get_version, finalize_onnx_artifact
    from .model import SIGLIP_MEAN, SIGLIP_STD

    res = int(imgsz or model.input_size)
    opset = 14 if opset is None else int(opset)
    if opset < 14:
        raise ValueError("LibreSigLIP2 ONNX export needs opset >= 14 (ViT attention).")
    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
        raise ValueError(
            f"LibreSigLIP2 export batch must be a positive integer, got {batch!r}."
        )

    vision = model.model.vision_model
    first_parameter = next(vision.parameters())
    original_device = first_parameter.device
    trace_device = _resolve_export_device(device, original_device)
    trace_dtype = first_parameter.dtype

    text_embeds = model._text_embeds.detach().to(trace_device, trace_dtype)
    scale = float(model.model.logit_scale.exp().detach().cpu())
    weight = scale * text_embeds  # [K, D]
    bias = model.model.logit_bias.detach().to(trace_device, trace_dtype).reshape(())

    training_states = {module: module.training for module in vision.modules()}

    if output is None:
        output = f"{model.FILENAME_PREFIX}{model.size}-cls.onnx"
    output = str(output)
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.zeros(
        batch,
        3,
        res,
        res,
        dtype=trace_dtype,
        device=trace_device,
    )
    dynamic_axes = {"images": {0: "batch"}, "logits": {0: "batch"}} if dynamic else None
    try:
        vision = vision.to(trace_device).eval()
        frozen = _FrozenSigLIP2Classifier(vision, weight, bias).eval()
        with torch.no_grad():
            torch.onnx.export(
                frozen,
                dummy,
                output,
                input_names=["images"],
                output_names=["logits"],
                opset_version=opset,
                dynamic_axes=dynamic_axes,
                verbose=bool(verbose),
                dynamo=False,
            )
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "libreyolo_version": _get_version(),
            "model_family": model.FAMILY,
            "size": model.size,
            "model_size": model.size,
            "task": "classify",
            "supported_tasks": json.dumps(["classify"]),
            "default_task": "classify",
            "nc": str(model.nb_classes),
            "nb_classes": str(model.nb_classes),
            "names": json.dumps(
                {str(key): value for key, value in model.names.items()}
            ),
            "imgsz": str(res),
            "imgsz_h": str(res),
            "imgsz_w": str(res),
            "precision": "fp16" if trace_dtype == torch.float16 else "fp32",
            "dynamic": str(bool(dynamic)),
            "half": str(trace_dtype == torch.float16),
            "classification_mean": json.dumps(list(SIGLIP_MEAN)),
            "classification_std": json.dumps(list(SIGLIP_STD)),
            "classification_crop_pct": "1.0",
            "classification_interpolation": "bilinear",
            "classification_square_resize": "true",
            "classification_activation": (
                "sigmoid" if model._multi_label else "softmax"
            ),
            "crop_pct": "1.0",
            "interpolation": "bilinear",
        }
        finalize_onnx_artifact(
            output,
            simplify=bool(simplify),
            dynamic=bool(dynamic),
            half=trace_dtype == torch.float16,
            metadata=metadata,
        )
    finally:
        vision.to(original_device)
        for module, training in training_states.items():
            module.training = training
    return output


def _resolve_export_device(
    requested: str | torch.device | int | None,
    current: torch.device,
) -> torch.device:
    """Resolve the tracing device without changing the owning model's device."""
    if requested is None or str(requested).lower() == "auto":
        return current
    if isinstance(requested, int) or (
        isinstance(requested, str) and requested.isdigit()
    ):
        requested = f"cuda:{requested}"
    return torch.device(requested)
