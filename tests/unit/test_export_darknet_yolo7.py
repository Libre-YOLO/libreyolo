"""Export-wrapper output contract for the Darknet families + YOLOv7.

The decode is baked into the export graph (a single ``(B, 4+nc, N)`` tensor), so
this pins the shape and score range the wrapper must produce.

Parity between an exported ONNX / TorchScript graph and this native wrapper
output is covered for all four families by ``test_darknet_edge_export_matrix.py``
(same assertion, at 96px instead of 416/640), so it is not repeated here.
"""

from __future__ import annotations

import pytest
import torch

from libreyolo import LibreYOLO2, LibreYOLO3, LibreYOLO4, LibreYOLO7
from libreyolo.models.darknet.export import DarknetExportWrapper
from libreyolo.models.yolo7.export import YOLO7ExportWrapper

pytestmark = pytest.mark.unit


def _wrapper(model):
    if model.FAMILY == "yolo7":
        return YOLO7ExportWrapper(model.model).eval()
    return DarknetExportWrapper(model.model).eval()


CASES = [
    (LibreYOLO2, "t", 416),
    (LibreYOLO3, "t", 416),
    (LibreYOLO4, "t", 416),
    (LibreYOLO7, "b", 640),
]


@pytest.mark.parametrize("cls,size,imgsz", CASES)
def test_export_wrapper_output_contract(cls, size, imgsz):
    model = cls(size=size, device="cpu")
    model.model.eval()
    x = torch.zeros(1, 3, imgsz, imgsz)
    with torch.no_grad():
        out = _wrapper(model)(x)
    # (B, 4+nc, N): boxes + per-class scores, N anchors flattened over heads
    assert out.shape[0] == 1
    assert out.shape[1] == 4 + model.nb_classes
    assert out.shape[2] > 0
    # scores are probabilities in [0, 1]
    scores = out[:, 4:, :]
    assert float(scores.min()) >= 0.0 and float(scores.max()) <= 1.0
