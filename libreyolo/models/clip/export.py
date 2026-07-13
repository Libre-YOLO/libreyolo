"""Frozen-class ONNX export for LibreCLIP.

Open-vocabulary export (two towers + a tokenizer) is awkward and out of scope
for v1. Instead we bake the *current* ``set_classes`` text embeddings into a
final linear projection so the graph is an ordinary ``[B, K]`` image classifier:

    logits = (logit_scale.exp() * text_embeds) @ L2norm(image_tower(x))

The exported model is fixed to the labels and input resolution at export time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.serialization import SCHEMA_VERSION


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
    from .model import CLIP_MEAN, CLIP_STD

    res = int(imgsz or model.input_size)
    opset = 14 if opset is None else int(opset)
    if opset < 14:
        raise ValueError("LibreCLIP ONNX export needs opset >= 14 (ViT attention).")
    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
        raise ValueError(
            f"LibreCLIP export batch must be a positive integer, got {batch!r}."
        )

    visual = model.model.visual
    first_parameter = next(visual.parameters())
    original_device = first_parameter.device
    trace_device = _resolve_export_device(device, original_device)
    trace_dtype = first_parameter.dtype

    text_embeds = model._text_embeds.detach().to(trace_device, trace_dtype)
    scale = float(model.model.logit_scale.exp().detach().cpu())
    weight = scale * text_embeds  # [K, D]

    training_states = {module: module.training for module in visual.modules()}

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
        visual = visual.to(trace_device).eval()
        frozen = _FrozenCLIPClassifier(visual, weight).eval()
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
                # Stable TorchScript exporter (no onnxscript dependency); the
                # frozen graph is a plain ViT + matmul that exports cleanly.
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
            "classification_mean": json.dumps(list(CLIP_MEAN)),
            "classification_std": json.dumps(list(CLIP_STD)),
            "classification_crop_pct": "1.0",
            "classification_interpolation": "bicubic",
            "classification_square_resize": "false",
            "classification_activation": "softmax",
            "crop_pct": "1.0",
            "interpolation": "bicubic",
        }
        finalize_onnx_artifact(
            output,
            simplify=bool(simplify),
            dynamic=bool(dynamic),
            half=trace_dtype == torch.float16,
            metadata=metadata,
        )
    finally:
        # Restore the tower to its original device even if export raised, so the
        # model instance stays usable for further predict()/_forward() calls.
        visual.to(original_device)
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
