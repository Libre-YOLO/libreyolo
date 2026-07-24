"""ONNX export for LibrePAGE.

The exported graph has three inputs — ``scene`` [1, 3, 512, 512],
``heads`` [N, 3, 256, 256], ``head_rects`` [N, 4] (scene-grid units, see
``utils.head_rects_grid``) — and two outputs, ``heatmap`` [N, 64, 64] and
``inout`` [N], both sigmoid probabilities. N (people per image) is a
dynamic axis. Preprocessing and heatmap decoding stay on the caller side,
identical to the PyTorch path (``utils.preprocess_scene_and_heads`` /
``utils.decode_heatmaps``).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn

from .nn import HEAD_SIZE, SCENE_SIZE

if TYPE_CHECKING:
    from .model import LibrePAGE


class _PageExportWrapper(nn.Module):
    """Flatten the (heatmap_logits, inout_logits) tuple into sigmoid outputs."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, scene, heads, head_rects):
        heatmap_logits, inout_logits = self.model(scene, heads, head_rects)
        return torch.sigmoid(heatmap_logits), torch.sigmoid(inout_logits)


def export_page_onnx(
    model: "LibrePAGE",
    output_path: Optional[str] = None,
    opset: int = 17,
    dynamic: bool = True,
    **kwargs,
) -> str:
    """Export a LibrePAGE model to ONNX and return the output path."""
    del kwargs  # detection-shaped export kwargs (imgsz, nms, ...) do not apply

    if output_path is None:
        source = getattr(model, "model_path", None)
        if source:
            output_path = str(Path(str(source)).with_suffix(".onnx"))
        else:
            output_path = f"{model.FILENAME_PREFIX}{model.size}.onnx"

    net = _PageExportWrapper(model.model).eval().cpu()
    scene = torch.zeros(1, 3, *SCENE_SIZE)
    heads = torch.zeros(2, 3, *HEAD_SIZE)
    head_rects = torch.tensor([[8.0, 8.0, 16.0, 16.0], [4.0, 20.0, 10.0, 28.0]])

    dynamic_axes = (
        {
            "heads": {0: "people"},
            "head_rects": {0: "people"},
            "heatmap": {0: "people"},
            "inout": {0: "people"},
        }
        if dynamic
        else None
    )

    with torch.no_grad():
        torch.onnx.export(
            net,
            (scene, heads, head_rects),
            output_path,
            input_names=["scene", "heads", "head_rects"],
            output_names=["heatmap", "inout"],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )
    # Restore the runtime device placement disturbed by the CPU export.
    model.model.to(model.device)
    return output_path
