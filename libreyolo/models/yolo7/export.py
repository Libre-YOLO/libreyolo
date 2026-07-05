"""Export wrapper for LibreYOLO7 (YOLOv7).

Bakes the anchor decode into the traced graph so every export format emits a
single ``(B, 4+nc, N)`` tensor (xyxy input-pixel boxes + per-class scores)
consumed by the shared backend decode (`_parse_yolo9`).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ...postprocess.yolo7 import decode_export


class YOLO7ExportWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.net = model  # YOLOv7Model
        self._strides = model.strides
        self._nc = model.num_classes
        self._n_heads = len(model.anchors)
        # Register per-head anchors (A, 2) as device-tracked buffers.
        for i, flat in enumerate(model.anchors):
            pairs = list(zip(flat[0::2], flat[1::2]))
            self.register_buffer(
                f"anchors_{i}", torch.tensor(pairs, dtype=torch.float32)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        anchor_tensors = [getattr(self, f"anchors_{i}") for i in range(self._n_heads)]
        return decode_export(self.net(x), anchor_tensors, self._strides, self._nc)
