"""Export wrapper for the Darknet families (YOLOv2/v3/v4).

Bakes the anchor-box decode into the traced graph so every export format
(ONNX/NCNN/TFLite/TRT/CoreML/OpenVINO) emits a single self-contained
``(B, 4+nc, N)`` tensor (xyxy input-pixel boxes + per-class scores) that the
shared backend decode (`_parse_yolo9`) consumes. No learned anchors need to
travel in runtime metadata.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ...postprocess.darknet_yolo import decode_export


class DarknetExportWrapper(nn.Module):
    def __init__(self, holder: nn.Module):
        super().__init__()
        self.net = holder.net  # DarknetNet
        self._specs = holder.net.detection_specs()
        # Register anchors as device-tracked buffers (not baked constants).
        for i, spec in enumerate(self._specs):
            self.register_buffer(
                f"anchors_{i}", torch.tensor(spec.anchors, dtype=torch.float32)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        anchor_tensors = [getattr(self, f"anchors_{i}") for i in range(len(self._specs))]
        return decode_export(self.net(x), self._specs, anchor_tensors)
