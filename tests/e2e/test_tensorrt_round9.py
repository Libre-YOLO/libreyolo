"""TensorRT FP32 parity for the two Round 9 dense promotions.

BiRefNet and FeyNobg were also executed at their native 1024 canvas. Both
exports reach TensorRT's ONNX parser and stop because the standard runtime does
not provide the required ModulatedDeformConv2d plugin. Their measured blocks
live in the support registry rather than as permanently failing E2E cases.
LingBotVision and ZipDepth also export, reload, and predict, but repeated
TensorRT builds cross their promotion thresholds; those measured gaps likewise
stay in the support registry.
"""

from __future__ import annotations

import pytest

from .conftest import requires_tensorrt
from .test_tensorrt_round8 import TensorRTRound8Case, _run_tensorrt_case

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.export_backend,
    pytest.mark.experimental_backend,
    pytest.mark.tensorrt,
    pytest.mark.trt,
]


ROUND9_TENSORRT_VALIDATED_CASES = (
    TensorRTRound8Case("LibreDINOv2", "n", "semantic", 518, 3),
    TensorRTRound8Case("LibreEoMT", "s", "semantic", 512, 3),
)


@requires_tensorrt
@pytest.mark.slow
@pytest.mark.parametrize(
    "case",
    ROUND9_TENSORRT_VALIDATED_CASES,
    ids=lambda case: case.class_name,
)
def test_tensorrt_round9_dense_raw_and_predict_parity(tmp_path, case):
    _run_tensorrt_case(tmp_path, case)
